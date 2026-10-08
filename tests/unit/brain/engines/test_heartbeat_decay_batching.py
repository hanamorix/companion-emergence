"""INC-7 — heartbeat decay batching, keyset cursor, start check, cross-process
guard (spec §5; criteria C9, C10, C17, C30).

Fixtures: synthetic only (F-small, generated in-test with a deterministic
seed per 1.5-criteria.md's fixture rule) — never touches a real persona's data directory.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.bridge import cli_throttle
from brain.bridge.provider import FakeProvider
from brain.bridge.supervisor import _heartbeat_and_felt_time
from brain.emotion.decay import apply_decay
from brain.emotion.state import EmotionalState
from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

_SEED = 20260927  # deterministic (F-small fixture rule)


@pytest.fixture(autouse=True)
def _reset_cli_throttle():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


@pytest.fixture(autouse=True)
def _safe_tier_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same guard test_heartbeat.py uses: keep dream/reflex/research/growth
    from constructing a real CLI provider against a bare tmp_path persona."""
    monkeypatch.setattr(
        "brain.engines.heartbeat.build_tier_provider",
        lambda *a, **k: FakeProvider(),
    )


def _seed_memory(store: MemoryStore, *, tenderness: float = 5.0, love: float = 9.0) -> Memory:
    """A memory with one never-decaying channel (love, decay_half_life_days
    is None) and one finite-half-life channel (tenderness) — so a real decay
    pass both leaves one value untouched and changes the other, matching
    test_heartbeat.py's own established fixture shape."""
    m = Memory.create_new(
        content="seed",
        memory_type="conversation",
        domain="us",
        emotions={"love": love, "tenderness": tenderness},
    )
    store.create(m)
    return m


def _engine(tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix) -> HeartbeatEngine:
    return HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=tmp_path / "hb_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
    )


def _expected_decay(emotions: dict[str, float], elapsed_seconds: float) -> dict[str, float]:
    """Independent oracle: the exact production decay primitives
    (EmotionalState + apply_decay), applied ONCE, outside the batching loop
    — the reference a resumed/interrupted pass must match bit-for-bit."""
    state = EmotionalState()
    for name, intensity in emotions.items():
        state.set(name, float(intensity))
    apply_decay(state, elapsed_seconds)
    return {name: val for name, val in state.emotions.items() if val > 0.0}


# ---------------------------------------------------------------------------
# C9(a) — interrupted pass resumes after the saved cursor, same tick time,
# no row decayed twice.
# ---------------------------------------------------------------------------


def test_c9a_interrupted_pass_resumes_without_double_decay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        mems = [_seed_memory(store, tenderness=5.0 + i * 0.1) for i in range(5)]
        pre_decay_emotions = {m.id: dict(m.emotions) for m in mems}

        engine.run_tick(trigger="open")  # first-ever tick: init only

        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        # Force exactly ONE row per batch, deterministically (no wall-clock
        # race): a budget of 0.0 trips the "elapsed >= budget" check right
        # after the very first row is examined, every time.
        monkeypatch.setattr("brain.engines.heartbeat.HEARTBEAT_DECAY_BATCH_BUDGET_S", 0.0)

        # Inject "database is locked" at the START of batch 2 (b >= 2, per
        # C9(a)'s own fail-first scenario) — batch 1 must already have
        # committed and saved its cursor before this fires.
        real_update_batch = MemoryStore.update_emotions_batch
        call_count = {"n": 0}

        def _flaky_update_batch(self, rows):  # noqa: ANN001
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise sqlite3.OperationalError("database is locked")
            return real_update_batch(self, rows)

        monkeypatch.setattr(MemoryStore, "update_emotions_batch", _flaky_update_batch)

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            engine.run_tick(trigger="close")

        # Cursor left set (pass genuinely interrupted); last_tick_at untouched.
        interrupted_state = HeartbeatState.load(engine.state_path)
        assert interrupted_state is not None
        assert interrupted_state.decay_cursor is not None
        assert interrupted_state.last_tick_at == state.last_tick_at
        saved_tick_at = interrupted_state.decay_cursor.tick_at

        # Exactly one row (batch 1) was actually committed before the failure.
        reloaded_after_failure = {m.id: store.get(m.id) for m in mems}
        decayed_so_far = [
            mid
            for mid, mem in reloaded_after_failure.items()
            if mem.emotions != pre_decay_emotions[mid]
        ]
        assert len(decayed_so_far) == 1

        # Restore normal writes; the NEXT tick resumes (same tick_at).
        monkeypatch.setattr(MemoryStore, "update_emotions_batch", real_update_batch)
        result = engine.run_tick(trigger="close")

        final_state = HeartbeatState.load(engine.state_path)
        assert final_state is not None
        assert final_state.decay_cursor is None, "pass must be complete after the retry"
        assert final_state.last_tick_at == saved_tick_at, (
            "last_tick_at must advance to the INTERRUPTED pass's own tick_at, "
            "not a fresh `now` from the retry"
        )
        assert result.dream_gated_reason == "resumed_decay_only"

        elapsed_used = (saved_tick_at - state.last_tick_at).total_seconds()
        for m in mems:
            reloaded = store.get(m.id)
            assert reloaded is not None
            expected = _expected_decay(pre_decay_emotions[m.id], elapsed_used)
            assert reloaded.emotions == pytest.approx(expected, abs=1e-12), (
                "resumed pass's final value must equal a SINGLE uninterrupted "
                "decay application at the same elapsed time — not a second, "
                "additional decay on top of the row batch 1 already committed"
            )
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# C9(b) — no pass starts while a reply is in flight; a reply starting
# mid-pass does not stop it.
# ---------------------------------------------------------------------------


def test_c9b_no_pass_starts_while_reply_in_flight(tmp_path: Path) -> None:
    """Exercises the REAL supervisor call site (`_heartbeat_and_felt_time`),
    which — unlike the other tests in this file — builds its OWN
    `HeartbeatEngine` against the canonical persona-dir paths
    (`_run_heartbeat_tick`, supervisor.py), so this test seeds those exact
    paths directly rather than using this file's `_engine()` helper."""
    from brain.bridge.events import EventBus

    persona_dir = tmp_path
    store = MemoryStore(persona_dir / "memories.db")
    m = _seed_memory(store)
    pre = dict(m.emotions)
    store.close()

    state_path = persona_dir / "heartbeat_state.json"
    fresh = HeartbeatState.fresh(trigger="open")
    fresh.last_tick_at = fresh.last_tick_at - timedelta(hours=48)
    fresh.save(state_path)

    try:
        cli_throttle.note_user_message()  # a reply is "in flight"
        event_bus = EventBus()
        result = _heartbeat_and_felt_time(
            persona_dir, FakeProvider(), event_bus, last_heartbeat_at=time.monotonic()
        )
        assert result is None, "must return early, doing nothing, while a reply is in flight"

        check_store = MemoryStore(persona_dir / "memories.db")
        reloaded = check_store.get(m.id)
        check_store.close()
        assert reloaded is not None
        assert reloaded.emotions == pre, "no decay may occur while a reply is in flight"
        unchanged_state = HeartbeatState.load(state_path)
        assert unchanged_state is not None
        assert unchanged_state.decay_cursor is None
        assert unchanged_state.last_tick_at == fresh.last_tick_at

        cli_throttle.note_reply_end()
        result2 = _heartbeat_and_felt_time(
            persona_dir, FakeProvider(), event_bus, last_heartbeat_at=time.monotonic()
        )
        assert result2 is not None
        check_store2 = MemoryStore(persona_dir / "memories.db")
        reloaded2 = check_store2.get(m.id)
        check_store2.close()
        assert reloaded2 is not None
        assert reloaded2.emotions != pre, "once no reply is in flight, the pass proceeds"
    finally:
        cli_throttle.reset()


def test_c9b_reply_starting_mid_pass_does_not_stop_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S21: once started, a heartbeat pass runs to the end even if a reply
    starts mid-pass (only the START is checked, and only by the supervisor's
    own caller — HeartbeatEngine.run_tick itself never re-checks)."""
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        mems = [_seed_memory(store, tenderness=5.0 + i * 0.1) for i in range(4)]
        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        monkeypatch.setattr("brain.engines.heartbeat.HEARTBEAT_DECAY_BATCH_BUDGET_S", 0.0)
        real_update_batch = MemoryStore.update_emotions_batch
        calls = {"n": 0}

        def _mark_inflight_mid_pass(self, rows):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] == 2:
                cli_throttle.note_user_message()
            return real_update_batch(self, rows)

        monkeypatch.setattr(MemoryStore, "update_emotions_batch", _mark_inflight_mid_pass)

        result = engine.run_tick(trigger="close")
        assert result.memories_decayed == len(mems)
        assert calls["n"] >= len(mems), "the pass must not stop early once started"
        for m in mems:
            reloaded = store.get(m.id)
            assert reloaded is not None
            assert reloaded.emotions.get("tenderness", 0.0) < m.emotions["tenderness"]
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# C10(a) — UPDATE count equals rows whose values actually changed.
# ---------------------------------------------------------------------------


def test_c10a_update_count_equals_changed_rows_only(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        changing = [_seed_memory(store, tenderness=5.0 + i * 0.1) for i in range(4)]
        unchanging = [
            Memory.create_new(
                content="seed", memory_type="conversation", domain="us",
                emotions={"love": 9.0},  # ONLY a never-decaying channel present
            )
            for _ in range(3)
        ]
        for mem in unchanging:
            store.create(mem)
        protected = _seed_memory(store, tenderness=5.0)
        store.update(protected.id, protected=True)

        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        update_statements: list[str] = []
        store._conn.set_trace_callback(  # noqa: SLF001 — test-only introspection
            lambda sql: update_statements.append(sql)
            if sql.strip().upper().startswith("UPDATE MEMORIES")
            else None
        )
        try:
            result = engine.run_tick(trigger="close")
        finally:
            store._conn.set_trace_callback(None)  # noqa: SLF001

        assert result.memories_decayed == len(changing)
        assert len(update_statements) == len(changing)

        for m in unchanging:
            reloaded = store.get(m.id)
            assert reloaded is not None
            assert reloaded.emotions == m.emotions
        reloaded_protected = store.get(protected.id)
        assert reloaded_protected is not None
        assert reloaded_protected.emotions == protected.emotions
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# C10(b) — one transaction per batch; schema unaffected; lock-hold bound.
# ---------------------------------------------------------------------------


def test_c10b_one_transaction_per_batch_and_schema_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        schema_before = store._conn.execute(  # noqa: SLF001
            "PRAGMA table_info(memories)"
        ).fetchall()

        mems = [_seed_memory(store, tenderness=5.0 + i * 0.1) for i in range(6)]
        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        # Force exactly 2 rows per batch (budget trips after row 2 of each
        # batch) so 6 changing rows span 3 batches — deterministic, no
        # wall-clock race.
        real_update_batch = MemoryStore.update_emotions_batch
        batch_sizes: list[int] = []

        def _record_batch(self, rows):  # noqa: ANN001
            batch_sizes.append(len(rows))
            return real_update_batch(self, rows)

        monkeypatch.setattr(MemoryStore, "update_emotions_batch", _record_batch)

        # Budget of 0 forces 1 row/batch deterministically (see C9(a));
        # 6 rows -> 6 single-row batches, still one commit per batch (C10b).
        monkeypatch.setattr("brain.engines.heartbeat.HEARTBEAT_DECAY_BATCH_BUDGET_S", 0.0)

        begin_commit_events: list[str] = []
        store._conn.set_trace_callback(  # noqa: SLF001
            lambda sql: begin_commit_events.append(sql.strip().upper())
            if sql.strip().upper() in ("BEGIN", "COMMIT")
            else None
        )
        try:
            engine.run_tick(trigger="close")
        finally:
            store._conn.set_trace_callback(None)  # noqa: SLF001

        assert len(batch_sizes) == len(mems), "one batch per row at budget=0.0"
        assert all(n == 1 for n in batch_sizes)
        # Exactly one BEGIN/COMMIT pair per batch commit that happened
        # (sqlite3's own implicit-transaction tracing may not emit an
        # explicit "BEGIN" token for a deferred transaction on every driver
        # version, so the authoritative check is COMMIT count == batch count).
        commit_count = sum(1 for e in begin_commit_events if e == "COMMIT")
        assert commit_count == len(mems)

        schema_after = store._conn.execute("PRAGMA table_info(memories)").fetchall()  # noqa: SLF001
        assert schema_after == schema_before, "no schema change (I2/I8)"
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# C17 — a chat-path write issued during an OPEN decay batch transaction
# succeeds after waiting (no "database is locked"), bounded ~ the budget.
# ---------------------------------------------------------------------------

_HOLDER_SCRIPT = """
import json, sqlite3, sys, time
db_path, ready_path, hold_seconds, memory_id, emotions_json = sys.argv[1:6]
hold_seconds = float(hold_seconds)
conn = sqlite3.connect(db_path, timeout=30.0)
conn.execute("PRAGMA busy_timeout = 30000")
conn.execute("BEGIN IMMEDIATE")
# Same SQL shape MemoryStore.update_emotions_batch issues for one row.
conn.execute(
    "UPDATE memories SET emotions_json = ?, "
    "peak_emotion_intensity = MAX(peak_emotion_intensity, ?) WHERE id = ?",
    (emotions_json, 9.0, memory_id),
)
with open(ready_path, "w") as f:
    f.write("1")
time.sleep(hold_seconds)
conn.commit()
conn.close()
"""

_HOLD_SECONDS = 2.0
_READY_WAIT_TIMEOUT_S = 15.0


def _wait_for_ready(ready_path: Path, holder: subprocess.Popen) -> None:
    deadline = time.monotonic() + _READY_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if ready_path.exists():
            return
        assert holder.poll() is None, "holder exited before signaling readiness"
        time.sleep(0.02)
    raise AssertionError("holder never signaled readiness")


def test_c17_chat_write_waits_for_open_decay_batch_then_succeeds(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.db"
    store = MemoryStore(db_path)
    try:
        m = _seed_memory(store, tenderness=5.0)
    finally:
        store.close()  # release before the holder subprocess opens its own connection

    ready_path = tmp_path / "holder.ready"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HOLDER_SCRIPT,
            str(db_path),
            str(ready_path),
            str(_HOLD_SECONDS),
            m.id,
            json.dumps({"love": 9.0, "tenderness": 4.0}),
        ]
    )
    try:
        _wait_for_ready(ready_path, holder)

        waiter_store = MemoryStore(db_path)
        try:
            start = time.monotonic()
            # A plain chat-path write on ANOTHER connection while the
            # holder's transaction (standing in for an open decay batch) is
            # still uncommitted — must succeed after waiting, not raise.
            waiter_store.update(m.id, importance=7.0)
            elapsed = time.monotonic() - start
        finally:
            waiter_store.close()

        assert elapsed >= _HOLD_SECONDS * 0.5, (
            "should have actually waited for the holder, not raced past it"
        )
        assert elapsed <= dev_constants.HEARTBEAT_DECAY_BATCH_BUDGET_S * 2 + 5.0, (
            "wait must be bounded — roughly the batch time budget, not the "
            "old 5s sqlite3 default's failure mode"
        )
    finally:
        returncode = holder.wait(timeout=_HOLD_SECONDS + 10.0)
        assert returncode == 0


# ---------------------------------------------------------------------------
# C30 — cross-process guard: a second concurrent pass skips instead of
# decaying from the same cursor.
# ---------------------------------------------------------------------------


def test_c30_second_concurrent_pass_skips(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        _seed_memory(store, tenderness=5.0)
        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        from brain.utils.file_lock import file_lock

        with file_lock(engine.state_path, blocking=False) as acquired:
            assert acquired
            # A second pass attempted while the first "holds" the lock (this
            # thread standing in for the concurrent holder) must skip.
            result = engine.run_tick(trigger="close")

        assert result.skipped_reason == "heartbeat_locked"
        assert result.memories_decayed == 0

        # Lock released — a real pass now runs normally.
        result2 = engine.run_tick(trigger="close")
        assert result2.skipped_reason is None
        assert result2.memories_decayed == 1
    finally:
        store.close()
        hebbian.close()


def test_c30_guard_is_inside_run_tick_not_bypassable_by_a_caller(tmp_path: Path) -> None:
    """C30's own text: the guard lives INSIDE run_tick so no caller can
    bypass it — proven here by calling run_tick directly (as cli.py and
    server.py's close-tick do) rather than through the supervisor, and
    confirming the same skip fires."""
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        engine.run_tick(trigger="open")
        from brain.utils.file_lock import file_lock

        with file_lock(engine.state_path, blocking=False):
            result = engine.run_tick(trigger="manual")
        assert result.skipped_reason == "heartbeat_locked"
    finally:
        store.close()
        hebbian.close()


def test_c30_state_is_read_only_after_the_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-6 red-team MAJOR, fixed: `HeartbeatState` must be loaded from
    disk only AFTER `run_tick` holds the C30 lock, never before — reading
    it before attempting the lock let a caller that won the lock only
    after a DIFFERENT process's pass had already completed and released it
    operate on a stale pre-lock snapshot, then overwrite that other
    process's freshly-completed state on save (a TOCTOU clobber). Proven
    here by asserting that a SEPARATE non-blocking lock attempt on the
    same sidecar fails at the exact moment `HeartbeatState.load_with_anomaly`
    is called — i.e. the real lock is already held by `run_tick` itself
    when the state read happens, not merely "around" it."""
    from brain.engines import heartbeat as heartbeat_module
    from brain.utils.file_lock import file_lock

    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        engine.run_tick(trigger="open")  # create heartbeat_state.json

        real_load = heartbeat_module.HeartbeatState.load_with_anomaly
        observations: list[bool] = []  # every call, not just the last

        def _spy_load(path):  # noqa: ANN001
            with file_lock(path, blocking=False) as could_also_acquire:
                observations.append(not could_also_acquire)
            return real_load(path)

        monkeypatch.setattr(
            heartbeat_module.HeartbeatState, "load_with_anomaly", staticmethod(_spy_load)
        )

        engine.run_tick(trigger="close")

        assert observations, "HeartbeatState.load_with_anomaly must be called at least once"
        assert all(observations), (
            "run_tick's own C30 lock must already be held at EVERY moment "
            "HeartbeatState is read from disk — a premature read before the "
            "lock is acquired is the exact TOCTOU this test guards against, "
            "even if a later, correctly-guarded read also happens"
        )
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# Regression guard: dry_run and the ordinary single-batch happy path are
# unaffected by the batching machinery (kept small/fast; the bulk of
# regression coverage is test_heartbeat.py's existing 79 tests, unmodified).
# ---------------------------------------------------------------------------


def test_corrupted_decay_cursor_reinitializes_instead_of_double_decaying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-6 red-team BLOCKER, fixed: a `decay_cursor` sub-field corrupted
    on disk (rest of the state file intact) must NOT silently fall back to
    "no cursor" while `last_tick_at` stays at its stale, pre-interruption
    value — that would start a FRESH pass with the same (already-applied)
    elapsed time and double-decay every row batch 1 already committed. It
    must instead be treated as a corrupt STATE (the same recovery every
    other malformed field on this dataclass already gets), which resets
    `last_tick_at` too — so the next tick's elapsed time is 0, not a
    second application of the same delta."""
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        m = _seed_memory(store, tenderness=5.0)
        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        # Force an interrupted pass so a real decay_cursor gets persisted,
        # then corrupt ONLY that sub-field on disk (simulating disk/write
        # corruption isolated to this new field, not the whole file).
        monkeypatch.setattr("brain.engines.heartbeat.HEARTBEAT_DECAY_BATCH_BUDGET_S", 0.0)
        real_update_batch = MemoryStore.update_emotions_batch
        calls = {"n": 0}

        def _fail_second_batch(self, rows):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] == 2:
                raise sqlite3.OperationalError("database is locked")
            return real_update_batch(self, rows)

        # Only ONE memory seeded above -> force a second (never-reached, so
        # harmless) memory to guarantee at least 2 batches would exist; add
        # one more row so batch 2 is real.
        m2 = _seed_memory(store, tenderness=6.0)
        monkeypatch.setattr(MemoryStore, "update_emotions_batch", _fail_second_batch)
        with pytest.raises(sqlite3.OperationalError):
            engine.run_tick(trigger="close")
        monkeypatch.setattr(MemoryStore, "update_emotions_batch", real_update_batch)

        interrupted = HeartbeatState.load(engine.state_path)
        assert interrupted is not None
        assert interrupted.decay_cursor is not None
        pre_corrupt_last_tick_at = interrupted.last_tick_at

        raw = json.loads(engine.state_path.read_text())
        raw["decay_cursor"]["tick_at"] = "not-a-valid-timestamp"
        engine.state_path.write_text(json.dumps(raw))
        for suffix in (".bak1", ".bak2", ".bak3"):
            bak = engine.state_path.with_name(engine.state_path.name + suffix)
            if bak.exists():
                bak.unlink()  # force past back-compat bak-rotation recovery too

        m1_before = store.get(m.id)
        m2_before = store.get(m2.id)
        assert m1_before is not None and m2_before is not None

        # Must NOT raise, and must NOT silently resume with a dropped
        # cursor + stale last_tick_at (the double-decay bug) — it may
        # either reinitialize (first_tick again) or otherwise recover, but
        # whatever it does, no row may be decayed by MORE than one
        # elapsed-time application relative to its pre-corruption value.
        engine.run_tick(trigger="close")

        recovered = HeartbeatState.load(engine.state_path)
        assert recovered is not None
        assert recovered.decay_cursor is None, "a corrupt cursor must not persist as-is"
        assert recovered.last_tick_at != pre_corrupt_last_tick_at or recovered.tick_count == 0, (
            "recovery must not silently keep the stale last_tick_at paired "
            "with a dropped cursor — that combination is exactly what "
            "causes a second, additional decay application"
        )

        m1_after = store.get(m.id)
        m2_after = store.get(m2.id)
        assert m1_after is not None and m2_after is not None
        # `m` sorts first by (created_at, id) so it was batch 1 — the row
        # ALREADY COMMITTED before the injected failure on batch 2 (`m2`).
        # It must not have decayed AGAIN relative to what batch 1 already
        # wrote — its post-batch-1 value must be unchanged by this
        # recovery tick (the exact reproduction the red-team found: it
        # used to keep decaying by a second elapsed-time application).
        assert m1_after.emotions == m1_before.emotions, (
            "the row batch 1 already committed must not be decayed a "
            "second time by the corrupted-cursor recovery tick"
        )
    finally:
        store.close()
        hebbian.close()


def test_dry_run_on_a_resumed_pass_never_touches_cursor_or_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-6 red-team MINOR: the resume path (`_resume_decay_only`) under
    `dry_run=True` had no direct test coverage. `_apply_emotion_decay`
    short-circuits to a no-op under dry_run before touching the cursor, so
    `_resume_decay_only`'s own `if not dry_run:` block (state/log writes)
    never runs — a resumed pass's saved cursor must survive a dry_run call
    completely untouched."""
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        m = _seed_memory(store, tenderness=5.0)
        m2 = _seed_memory(store, tenderness=6.0)
        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        monkeypatch.setattr("brain.engines.heartbeat.HEARTBEAT_DECAY_BATCH_BUDGET_S", 0.0)
        real_update_batch = MemoryStore.update_emotions_batch
        calls = {"n": 0}

        def _fail_second_batch(self, rows):  # noqa: ANN001
            calls["n"] += 1
            if calls["n"] == 2:
                raise sqlite3.OperationalError("database is locked")
            return real_update_batch(self, rows)

        monkeypatch.setattr(MemoryStore, "update_emotions_batch", _fail_second_batch)
        with pytest.raises(sqlite3.OperationalError):
            engine.run_tick(trigger="close")
        monkeypatch.setattr(MemoryStore, "update_emotions_batch", real_update_batch)

        interrupted = HeartbeatState.load(engine.state_path)
        assert interrupted is not None
        assert interrupted.decay_cursor is not None
        m1_mid = store.get(m.id)
        m2_mid = store.get(m2.id)

        result = engine.run_tick(trigger="close", dry_run=True)
        assert result.memories_decayed == 0
        assert result.dream_gated_reason == "resumed_decay_only"

        after_dry_run = HeartbeatState.load(engine.state_path)
        assert after_dry_run is not None
        assert after_dry_run.decay_cursor == interrupted.decay_cursor, (
            "dry_run must not clear or advance a resumed pass's saved cursor"
        )
        assert after_dry_run.last_tick_at == interrupted.last_tick_at
        assert after_dry_run.tick_count == interrupted.tick_count
        assert store.get(m.id).emotions == m1_mid.emotions
        assert store.get(m2.id).emotions == m2_mid.emotions

        # A REAL (non-dry) retry afterwards still completes normally.
        final = engine.run_tick(trigger="close")
        assert final.dream_gated_reason == "resumed_decay_only"
        final_state = HeartbeatState.load(engine.state_path)
        assert final_state is not None
        assert final_state.decay_cursor is None
    finally:
        store.close()
        hebbian.close()


def test_dry_run_never_touches_cursor_or_store(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        m = _seed_memory(store, tenderness=5.0)
        pre = dict(m.emotions)
        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        result = engine.run_tick(trigger="close", dry_run=True)
        assert result.memories_decayed == 0
        reloaded = store.get(m.id)
        assert reloaded is not None
        assert reloaded.emotions == pre

        post_state = HeartbeatState.load(engine.state_path)
        assert post_state is not None
        assert post_state.decay_cursor is None
        assert post_state.last_tick_at == state.last_tick_at, "dry_run must not advance state"
    finally:
        store.close()
        hebbian.close()
