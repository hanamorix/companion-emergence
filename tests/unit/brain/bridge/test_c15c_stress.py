"""ram-spike-fix INC-10 — C15(c): heartbeat passes and gated-job items
driven for >= 60s on F-bob20k with concurrent chat-path writes from another
thread -> zero "database is locked" (2-plan §4.3/§4.2's whole point: the
30s busy timeout + batched decay removes the O16 starvation pattern even
under real, item-level, multi-minute background activity).

Stage-6 red-team MAJOR, addressed: C15(a)/(b) (job-granularity ordering and
heartbeat/job mutual exclusion) were proven at INC-9; this file proves
C15(c) specifically — the ITEM-level interleaving this increment (INC-10)
introduces, under real concurrent chat-path writes, for the full 60s+ the
criterion names.

Opt-in (#421): runs only with KINDLED_RUN_STRESS=1, because it takes over a
minute; test.yml's python-313-bridge job sets it. The database is the real
F-bob20k fixture (~20k rows, 107MB, not committed) when KINDLED_F_BOB20K
names it, else a synthetic 20k-row memories.db built in a few seconds. (The
fixture path used to be derived from a worktree layout that no longer
exists, so the test skipped everywhere.) Runs `run_folded`
in a background thread for >= 60s wall-clock with the heartbeat and the
maintenance (forgetting) job firing continuously, while a SEPARATE thread
hammers `store.bump_recall` on a second connection to the SAME memories.db
the whole time — the real chat-path write this criterion means by
"concurrent chat-path writes."

Liveness (#437, #445): the test used to assert only "no write errors, no
database-is-locked log hits", so a loop that never ran, died at startup or did
no useful work passed, and a thread that failed to stop was silently left
running. It now also asserts, for the supervisor thread: it raised nothing, was
alive at the end of the window, and is not alive after `stop_event` plus a
bounded join (no help from the test: the supervisor itself must stop a gated
pass at its next item); and that the loop did real work in the window: a
heartbeat that decayed memories with no reflex/growth error, forgetting wrote
rows, a `supervisor_tick` (only "the loop iterates"), and no exception swallowed
by the loop's own loggers. The writer thread is held to the same standard (no
error, alive, joined, at least one successful `bump_recall`). The persona is
seeded with a felt-time age so the first forgetting pass is not the cold-start
all-exempt no-op. The whole body is `_stress_body`; the default-suite tests at
the bottom run the same body against stand-in loops (no-op, crash, ignores
stop, silent, ticks only, ...) and require the matching failure, so every new
assertion is proven able to fail, and deleting one fails the default suite.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import brain.bridge.provider as provider_mod
from brain.bridge import cli_throttle, job_progress
from brain.bridge.events import EventBus
from brain.felt_time.state import FeltTimeState, load_or_recover, persist

_FIXTURE_ENV = os.environ.get("KINDLED_F_BOB20K")


def _build_synthetic_bob20k(db_path: Path, n: int = 20_000) -> None:
    """20k memories spread over a year with nonzero emotions, so the
    heartbeat's batched decay and the forgetting pass both have real work
    on every row, as they did on F-bob20k. (`n` is only lowered by the
    default-suite stand-in tests, which need a database but not its size.)"""
    from brain.memory.store import Memory, MemoryStore

    now = datetime.now(UTC)
    store = MemoryStore(str(db_path), integrity_check=False)
    try:
        for i in range(n):
            mem = Memory.create_new(
                f"memory {i}: an ordinary moment worth keeping",
                "conversation",
                "us",
                emotions={"warmth": (i % 9) + 1.0, "longing": (i % 4) * 0.5},
            )
            store.create(dataclasses.replace(mem, created_at=now - timedelta(hours=i % 8760)))
    finally:
        store.close()


_STRESS_DURATION_S = 60.0
_STRESS_ROWS = 20_000
# The bridge's own shutdown budget: server.py joins the supervisor thread with
# `join(timeout=180.0)`. A stop that arrives while a heartbeat is running waits
# for it (a heartbeat over 20k rows takes many seconds), so the old 30 s was
# not a safe bound on a slow host; the production budget is.
_SHUTDOWN_JOIN_TIMEOUT_S = 180.0
# Felt-time age written into the persona before the loop starts. At cold start
# (age 0) `policy.is_exempt` exempts EVERY memory, so the first forgetting pass
# would score and write nothing; with a lived age the pass does real work.
_SEEDED_LIVED_AGE_HOURS = 2000.0


class _LockErrorCapture(logging.Handler):
    """Captures any log record whose message OR attached exception mentions
    a database lock — the observable signature of the O16 starvation
    pattern this fix removes, across every logger in the process
    (heartbeat, forgetting, store, supervisor).

    Round-2 red-team MAJOR, fixed: `record.getMessage()` alone is NOT
    enough — `brain/bridge/supervisor.py`'s heartbeat/forgetting call sites
    catch with a bare `except Exception: logger.exception("<generic
    label>")`, which never interpolates the caught exception's own text
    into the message (verified: `LogRecord.getMessage()` returns only the
    literal format string, never `exc_info`/`exc_text`). A real "database
    is locked" raised on either of those paths would previously go
    completely undetected by this oracle. `logger.exception(...)` attaches
    `exc_info` to the record; format it (via `self.format`, which renders
    the traceback + `repr(exception)` through the handler's own formatter)
    and scan THAT too, not just the bare message.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.hits: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        haystacks = [record.getMessage()]
        if record.exc_info is not None:
            # self.format() renders the message + the formatted traceback
            # (which includes str(exception), e.g. "OperationalError:
            # database is locked") through this handler's own formatter.
            haystacks.append(self.format(record))
        text = " ".join(haystacks).lower()
        if "database is locked" in text or ("locked" in text and "database" in text):
            # Round-3 red-team MINOR, fixed: record the actual matched text
            # (including the exception's own "database is locked" message
            # when that's what fired), not just the generic log label --
            # a future failure message should be debuggable at a glance.
            self.hits.append(text if len(haystacks) > 1 else record.getMessage())


class _SwallowedErrorCapture(logging.Handler):
    """Records any WARNING-or-higher log record that carries an exception
    (`logger.exception(...)`, `exc_info=...`) from the loggers the supervisor
    loop and the forgetting pass own. `run_folded` swallows every per-section
    error into a log line, so without this a loop that fails every job with a
    non-lock error (disk I/O, malformed database, ...) but keeps ticking would
    pass. `_LockErrorCapture` above only matches lock errors."""

    _PREFIXES = ("brain.bridge.", "brain.forgetting")

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.hits: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING or record.exc_info is None:
            return
        if not record.name.startswith(self._PREFIXES):
            return
        self.hits.append(f"{record.name}: {record.getMessage()}")


class _RecordingBus(EventBus):
    """The event bus the loop publishes to, recording every event. The loop's
    work is observable here with no product hook: `supervisor_tick` per pass,
    `heartbeat_tick` per finished heartbeat, `forgetting_pass` per forgetting
    pass or pause."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(event)

    def of_type(self, event_type: str) -> list[dict]:
        return [e for e in list(self.events) if e.get("type") == event_type]


class _ThreadRun:
    """A daemon thread whose exception is captured instead of lost to the
    threading excepthook, with the liveness facts the assertions need."""

    def __init__(self, target: Callable[..., object], kwargs: dict) -> None:
        self.error: BaseException | None = None
        self.alive_at_window_end: bool | None = None
        self.joined: bool | None = None
        self._target = target
        self._kwargs = kwargs
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self._target(**self._kwargs)
        except BaseException as exc:  # noqa: BLE001 — record, the assertions report it
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def sample_alive(self) -> None:
        self.alive_at_window_end = self.thread.is_alive()

    def join(self, timeout_s: float) -> None:
        self.thread.join(timeout=timeout_s)
        self.joined = not self.thread.is_alive()


def _chat_writer(
    *,
    persona_dir: Path,
    stop: threading.Event,
    counter: list[int],
    write_errors: list[Exception],
) -> None:
    """A real chat-path write on a SECOND connection to the same
    memories.db, hammering continuously for the whole stress window —
    this is what C15(c) means by "concurrent chat-path writes". A failure to
    open the store or to find a row now raises (the thread wrapper records it)
    instead of silently doing nothing, and every successful bump is counted."""
    from brain.memory.store import MemoryStore

    writer_store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    try:
        row = writer_store._conn.execute("SELECT id FROM memories LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("the stress database has no memory row for the writer to bump")
        memory_id = row["id"]
        while not stop.is_set():
            try:
                writer_store.bump_recall(memory_id, 0.01)
                counter[0] += 1
            except Exception as exc:  # noqa: BLE001 — record, don't crash the writer thread
                write_errors.append(exc)
            time.sleep(0.01)
    finally:
        writer_store.close()


@dataclasses.dataclass
class _StressResult:
    loop: _ThreadRun
    writer: _ThreadRun
    bus: _RecordingBus
    lock_capture: _LockErrorCapture
    swallowed: _SwallowedErrorCapture
    write_errors: list[Exception]
    bumps: int
    join_timeout_s: float
    non_active_rows: int | None = None
    forgetting_cursor: dict = dataclasses.field(default_factory=dict)
    forgetting_state_file: bool = False
    oracle_error: str | None = None
    loop_join_s: float = 0.0
    subprocess_calls: list[str] = dataclasses.field(default_factory=list)


def _read_window_end_oracle(persona_dir: Path, result: _StressResult) -> None:
    """What forgetting left behind, read while the loop and the writer are
    still running: a row no longer `active` (a fade write landed), and the
    resume cursor / counters file (item-level work was in flight). Read through
    a separate query-only connection; an unreadable database is reported as an
    oracle error, distinct from a lock hit in the loop."""
    try:
        con = sqlite3.connect(str(persona_dir / "memories.db"), timeout=30.0)
        try:
            con.execute("PRAGMA query_only=ON")
            result.non_active_rows = con.execute(
                "SELECT count(*) FROM memories WHERE state != 'active'"
            ).fetchone()[0]
        finally:
            con.close()
        result.forgetting_cursor = job_progress.load_progress(persona_dir, "forgetting")
        result.forgetting_state_file = (persona_dir / "forgetting_state.json").exists()
    except Exception as exc:  # noqa: BLE001
        result.oracle_error = repr(exc)


def _run_stress_window(
    persona_dir: Path,
    *,
    loop_target: Callable[..., object],
    loop_kwargs: dict,
    writer_target: Callable[..., object],
    duration_s: float,
    join_timeout_s: float,
    writer_join_timeout_s: float = 10.0,
    cleanup_join_timeout_s: float = 10.0,
    settle: Callable[[_ThreadRun, _ThreadRun, list[int]], None] | None = None,
) -> _StressResult:
    """Run the loop and the writer side by side for `duration_s` of real time,
    sample liveness and the window-end oracle, then stop both and join them
    with a bounded wait. Nothing here helps the loop stop: only `stop_event`.
    If the loop is still alive after the bounded join the result is recorded
    first, and only then is chat marked active (cleanup only) so a red run
    does not leave a live thread in the pytest process."""
    lock_capture = _LockErrorCapture()
    swallowed = _SwallowedErrorCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(lock_capture)
    root_logger.addHandler(swallowed)

    write_errors: list[Exception] = []
    counter = [0]
    stop_writer = threading.Event()
    stop_event: threading.Event = loop_kwargs["stop_event"]
    writer = _ThreadRun(
        writer_target,
        {
            "persona_dir": persona_dir,
            "stop": stop_writer,
            "counter": counter,
            "write_errors": write_errors,
        },
    )
    loop = _ThreadRun(loop_target, loop_kwargs)
    result = _StressResult(
        loop=loop,
        writer=writer,
        bus=loop_kwargs["event_bus"],
        lock_capture=lock_capture,
        swallowed=swallowed,
        write_errors=write_errors,
        bumps=0,
        join_timeout_s=join_timeout_s,
    )
    writer.start()
    loop.start()
    try:
        # Drive heartbeat + gated-job items for the full window the
        # criterion names, real wall-clock time (not a fake/injected
        # clock — this is a stress test of real contention).
        time.sleep(duration_s)
        if settle is not None:
            settle(loop, writer, counter)
        writer.sample_alive()
        loop.sample_alive()
        _read_window_end_oracle(persona_dir, result)
    finally:
        stop_writer.set()
        stop_event.set()
        joining_since = time.monotonic()
        loop.join(join_timeout_s)
        result.loop_join_s = time.monotonic() - joining_since
        writer.join(writer_join_timeout_s)
        if not loop.joined:
            cli_throttle.note_user_message()  # cleanup only, after the result is recorded
            loop.thread.join(timeout=cleanup_join_timeout_s)
        root_logger.removeHandler(lock_capture)
        root_logger.removeHandler(swallowed)
        result.bumps = counter[0]
    return result


def _assert_stress_result(result: _StressResult) -> None:
    write_errors = result.write_errors
    lock_capture = result.lock_capture
    # The two original assertions, unchanged, first.
    assert not write_errors, (
        f"the concurrent chat-path writer hit {len(write_errors)} exception(s), "
        f"first: {write_errors[0] if write_errors else None}"
    )
    assert lock_capture.hits == [], (
        f"'database is locked' (or similar) logged during the {_STRESS_DURATION_S}s "
        f"stress window: {lock_capture.hits[:5]}"
    )

    loop, writer, bus = result.loop, result.writer, result.bus
    assert result.subprocess_calls == [], (
        "a real process was started during the stress window (the real Claude CLI "
        f"provider?): {result.subprocess_calls[:3]}"
    )
    assert loop.error is None, f"the supervisor loop thread raised {loop.error!r}"
    assert loop.alive_at_window_end, (
        "the supervisor loop thread was not alive at the end of the window"
    )
    assert loop.joined, (
        f"the supervisor loop thread did not join within {result.join_timeout_s}s after "
        "stop_event was set (a gated pass did not stop at its next item)"
    )
    assert bus.of_type("supervisor_tick"), (
        "no supervisor_tick event: the loop never completed a pass"
    )
    heartbeats = bus.of_type("heartbeat_tick")
    assert heartbeats, "no heartbeat_tick event: no heartbeat completed in the window"
    assert any((hb.get("memories_decayed") or 0) > 0 for hb in heartbeats), (
        "no heartbeat_tick decayed any memory: the heartbeat's write path did not run"
    )
    failed = [hb for hb in heartbeats if hb.get("reflex_error") or hb.get("growth_error")]
    assert not failed, (
        f"{len(failed)} heartbeat_tick event(s) carried a reflex_error/growth_error, "
        f"first: {failed[0].get('reflex_error') or failed[0].get('growth_error')}"
    )
    assert result.oracle_error is None, (
        f"could not read the forgetting oracle at the end of the window: {result.oracle_error}"
    )
    assert result.non_active_rows, (
        "forgetting wrote no row in the window (no memory left the 'active' state): "
        "the maintenance pass did no item-level work"
    )
    assert result.forgetting_cursor or result.forgetting_state_file, (
        "forgetting left neither a resume cursor nor forgetting_state.json: "
        "no item was processed in the window"
    )
    assert result.swallowed.hits == [], (
        "swallowed exception logged by the loop during the window: "
        f"{result.swallowed.hits[:5]}"
    )
    assert writer.error is None, f"the writer thread raised {writer.error!r}"
    assert writer.alive_at_window_end, (
        "the writer thread was not alive at the end of the window"
    )
    assert writer.joined, "the writer thread did not join after it was told to stop"
    assert result.bumps >= 1, "the writer made no successful bump_recall in the window"


def _subprocess_blocker(calls: list[str]) -> Callable[..., object]:
    """A stand-in for `subprocess.Popen` that records the command and refuses.
    The stress test must never start a real process: the heartbeat, the soul
    review and every other tick build their provider from the persona's
    `persona_config.json`, and with the default provider that is the real Claude
    CLI (a subprocess per call). On a machine with `claude` installed that
    silently works; on a runner without it every heartbeat raised and the test
    could not pass. `persona_config.json` now says `fake`, and this blocker makes
    any regression fail loudly (see `_assert_stress_result`)."""

    def _blocked(*args: object, **_kwargs: object) -> object:
        command = args[0] if args else "?"
        calls.append(" ".join(map(str, command)) if isinstance(command, (list, tuple)) else str(command))
        raise OSError("blocked: the stress test must not start a real process (no real Claude CLI)")

    return _blocked


def _stress_body(
    tmp_path: Path,
    *,
    loop_target: Callable[..., object],
    duration_s: float,
    rows: int,
    join_timeout_s: float,
    writer_target: Callable[..., object] = _chat_writer,
    writer_join_timeout_s: float = 10.0,
    cleanup_join_timeout_s: float = 10.0,
    settle: Callable[[_ThreadRun, _ThreadRun, list[int]], None] | None = None,
) -> _StressResult:
    """The whole stress test, with the loop injectable: set up the persona,
    run the window, assert. The opt-in test calls it with the real
    `run_folded`; the default-suite tests call it with stand-in loops."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    if _FIXTURE_ENV and rows == _STRESS_ROWS:
        # I11: copy into tmp_path, never open the fixture itself for write.
        shutil.copy(_FIXTURE_ENV, persona_dir / "memories.db")
    else:
        _build_synthetic_bob20k(persona_dir / "memories.db", rows)
    persist(FeltTimeState(lived_age_hours=_SEEDED_LIVED_AGE_HOURS), persona_dir)
    # Every tick builds its provider from this file (`build_tier_provider`), not
    # from the `provider=` argument: without it the default is the real Claude
    # CLI. `searcher: noop` keeps the heartbeat's research step off the network.
    (persona_dir / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}', encoding="utf-8"
    )

    loop_kwargs = {
        "stop_event": threading.Event(),
        "persona_dir": persona_dir,
        "provider": provider_mod.FakeProvider(),
        "event_bus": _RecordingBus(),
        "tick_interval_s": 0.05,
        "heartbeat_interval_s": 1.0,
        "soul_review_interval_s": 0.0,  # maintenance (forgetting) due every pass
        "finalize_interval_s": None,
        "log_rotation_interval_s": None,
        "initiate_review_interval_s": None,
        "voice_reflection_interval_s": None,
        "self_model_interval_s": None,
        "compaction_interval_s": None,
        "calibration_interval_s": None,
        "interest_sweep_interval_s": None,
        "judge_selftune_interval_s": None,
        "clustering_interval_s": None,
        "vocab_repair_interval_s": None,
        "pending_reconcile_interval_s": None,
        "maker_enabled": False,
        "notes_enabled": False,
        "kindled_link_enabled": False,
    }
    subprocess_calls: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "Popen", _subprocess_blocker(subprocess_calls))
        result = _run_stress_window(
            persona_dir,
            loop_target=loop_target,
            loop_kwargs=loop_kwargs,
            writer_target=writer_target,
            duration_s=duration_s,
            join_timeout_s=join_timeout_s,
            writer_join_timeout_s=writer_join_timeout_s,
            cleanup_join_timeout_s=cleanup_join_timeout_s,
            settle=settle,
        )
    result.subprocess_calls = subprocess_calls
    _assert_stress_result(result)
    return result


@pytest.mark.skipif(
    os.environ.get("KINDLED_RUN_STRESS") != "1",
    reason="60s stress test, opt-in: set KINDLED_RUN_STRESS=1 (run by test.yml's python-313-bridge job; #421)",
)
def test_c15c_heartbeat_and_gated_items_60s_with_concurrent_writes_no_lock_errors(
    tmp_path: Path,
) -> None:
    from brain.bridge.supervisor import run_folded

    result = _stress_body(
        tmp_path,
        loop_target=run_folded,
        duration_s=_STRESS_DURATION_S,
        rows=_STRESS_ROWS,
        join_timeout_s=_SHUTDOWN_JOIN_TIMEOUT_S,
    )
    # Visible with `pytest -s` / `-rP`: what the window actually did.
    bus = result.bus
    heartbeats = bus.of_type("heartbeat_tick")
    print(
        f"[c15c-stress] supervisor_tick={len(bus.of_type('supervisor_tick'))} "
        f"heartbeat_tick={len(heartbeats)} "
        f"decayed={[hb.get('memories_decayed') for hb in heartbeats]} "
        f"non_active_rows={result.non_active_rows} "
        f"cursor={bool(result.forgetting_cursor)} state_file={result.forgetting_state_file} "
        f"bumps={result.bumps} loop_join_s={result.loop_join_s:.2f}"
    )


# ---------------------------------------------------------------------------
# Default-suite tests (no env gate, 5-row database, ~0.3 s windows).
#
# They run the SAME `_stress_body` the opt-in test runs, with a stand-in for
# `run_folded` (and sometimes for the writer), and require the failure each new
# assertion exists to produce. A stand-in that is correct passes. Deleting any
# assertion, the felt-time seed or the join from `_stress_body` makes one of
# these fail, so the liveness checks cannot quietly rot (#437 was exactly
# that: a test whose oracle could not fail).
# ---------------------------------------------------------------------------

_FAST = {"duration_s": 0.3, "rows": 5, "join_timeout_s": 1.0}
_FAST_TEARDOWN = {"writer_join_timeout_s": 1.0, "cleanup_join_timeout_s": 0.1}


def _after_ready(
    ready: threading.Event, *, wait_for_bump: bool = True
) -> Callable[[_ThreadRun, _ThreadRun, list[int]], None]:
    """Wait for the stand-in loop to say it has published, and (unless the
    writer is the thing under test) for the writer to have made its first
    successful `bump_recall`, instead of trusting a bare short sleep on a
    loaded runner."""

    def _settle(_loop: _ThreadRun, _writer: _ThreadRun, counter: list[int]) -> None:
        assert ready.wait(5.0), "the stand-in loop never became ready"
        if wait_for_bump:
            deadline = time.monotonic() + 5.0
            while counter[0] < 1 and time.monotonic() < deadline:
                time.sleep(0.005)

    return _settle


def _after_loop_ends(loop: _ThreadRun, _writer: _ThreadRun, _counter: list[int]) -> None:
    """For a stand-in loop that is meant to be dead at the sample."""
    loop.thread.join(timeout=5.0)


def _after_writer_ends(
    ready: threading.Event,
) -> Callable[[_ThreadRun, _ThreadRun, list[int]], None]:
    """For a stand-in writer that is meant to be dead at the sample."""

    def _settle(_loop: _ThreadRun, writer: _ThreadRun, _counter: list[int]) -> None:
        assert ready.wait(5.0), "the stand-in loop never became ready"
        writer.thread.join(timeout=5.0)

    return _settle


def _stand_in(
    ready: threading.Event,
    *,
    release: threading.Event | None = None,
    crash: bool = False,
    noop: bool = False,
    ignores_stop: bool = False,
    tick: bool = True,
    heartbeats: tuple[dict, ...] = ({"memories_decayed": 1},),
    cursor: bool = True,
    state_file: bool = False,
    fade_row: bool = True,
    swallow_from: str | None = None,
    spawn_process: bool = False,
) -> Callable[..., None]:
    """A stand-in for `run_folded`: publishes what the real loop publishes and
    leaves what forgetting leaves, according to the flags, then waits for
    `stop_event`. The default flags describe a healthy loop."""

    def _target(**kwargs) -> None:
        persona_dir: Path = kwargs["persona_dir"]
        bus: EventBus = kwargs["event_bus"]
        stop: threading.Event = kwargs["stop_event"]
        if noop:
            return
        if crash:
            raise RuntimeError("stand-in loop crashed at startup")
        # The seed _stress_body must write: cold-start state exempts every memory.
        state, _recovered = load_or_recover(persona_dir)
        if state.lived_age_hours != _SEEDED_LIVED_AGE_HOURS:
            raise AssertionError(f"felt-time seed missing: lived_age_hours={state.lived_age_hours}")
        config = json.loads((persona_dir / "persona_config.json").read_text(encoding="utf-8"))
        if config.get("provider") != "fake":
            raise AssertionError(f"persona_config.json does not select the fake provider: {config}")
        if spawn_process:
            try:
                subprocess.run(["claude", "--version"], check=False)  # the blocker refuses
            except OSError:
                pass
        if tick:
            bus.publish({"type": "supervisor_tick"})
        for hb in heartbeats:
            bus.publish({"type": "heartbeat_tick", "reflex_error": None, "growth_error": None, **hb})
        if cursor:
            job_progress.save_progress(persona_dir, "forgetting", {"last_id": "stand-in"})
        if state_file:
            (persona_dir / "forgetting_state.json").write_text("{}", encoding="utf-8")
        if fade_row:
            con = sqlite3.connect(str(persona_dir / "memories.db"), timeout=30.0)
            try:
                con.execute(
                    "UPDATE memories SET state='fading' "
                    "WHERE id=(SELECT id FROM memories LIMIT 1)"
                )
                con.commit()
            finally:
                con.close()
        if swallow_from is not None:
            try:
                raise ValueError("stand-in swallowed error")
            except ValueError:
                logging.getLogger(swallow_from).exception("stand-in swallowed an exception")
        ready.set()
        if ignores_stop:
            assert release is not None
            release.wait(30.0)
        else:
            stop.wait(30.0)

    return _target


def _idle_writer(*, persona_dir, stop, counter, write_errors) -> None:  # noqa: ANN001
    """Alive until told to stop, but never writes."""
    stop.wait(30.0)


def _crashing_writer(*, persona_dir, stop, counter, write_errors) -> None:  # noqa: ANN001
    raise RuntimeError("stand-in writer could not open the store")


def _returning_writer(*, persona_dir, stop, counter, write_errors) -> None:  # noqa: ANN001
    return


def _writer_ignoring_stop(release: threading.Event) -> Callable[..., None]:
    def _target(*, persona_dir, stop, counter, write_errors) -> None:  # noqa: ANN001
        counter[0] += 1
        release.wait(30.0)

    return _target


def test_stand_in_healthy_loop_passes_every_assertion(tmp_path: Path) -> None:
    """Positive control: without it the cases below could be 'always fails'."""
    ready = threading.Event()
    result = _stress_body(
        tmp_path, loop_target=_stand_in(ready), settle=_after_ready(ready), **_FAST,
        **_FAST_TEARDOWN,
    )
    assert result.loop.joined and result.writer.joined


def test_stand_in_clean_finish_shape_passes(tmp_path: Path) -> None:
    """A forgetting pass that finished in the window clears its cursor but
    leaves forgetting_state.json and a non-active row: that is real work."""
    ready = threading.Event()
    _stress_body(
        tmp_path,
        loop_target=_stand_in(ready, cursor=False, state_file=True),
        settle=_after_ready(ready),
        **_FAST,
        **_FAST_TEARDOWN,
    )


@pytest.mark.parametrize(
    ("name", "kwargs", "settle_kind", "phrase"),
    [
        ("loop_noop", {"noop": True}, "dead", "supervisor loop thread was not alive"),
        ("loop_crash", {"crash": True}, "dead", "supervisor loop thread raised"),
        ("silent_loop", {"tick": False, "heartbeats": (), "cursor": False, "fade_row": False},
         "ready", "supervisor_tick"),
        ("ticks_only", {"heartbeats": (), "cursor": False, "fade_row": False},
         "ready", "heartbeat_tick"),
        ("heartbeat_decays_nothing", {"heartbeats": ({"memories_decayed": 0},)},
         "ready", "decayed any memory"),
        ("heartbeat_reports_reflex_error", {"heartbeats": ({"memories_decayed": 1,
                                                            "reflex_error": "boom"},)},
         "ready", "reflex_error"),
        ("heartbeat_reports_growth_error", {"heartbeats": ({"memories_decayed": 1,
                                                            "growth_error": "boom"},)},
         "ready", "growth_error"),
        ("all_exempt_shape", {"cursor": False, "fade_row": False},
         "ready", "forgetting wrote no row"),
        ("cursor_but_no_row_written", {"fade_row": False},
         "ready", "forgetting wrote no row"),
        ("row_written_but_no_cursor_or_state", {"cursor": False},
         "ready", "neither a resume cursor"),
        ("loop_starts_a_real_process", {"spawn_process": True},
         "ready", "real process was started"),
        ("loop_swallows_an_exception", {"swallow_from": "brain.bridge.supervisor"},
         "ready", "swallowed exception"),
        ("forgetting_swallows_an_exception", {"swallow_from": "brain.forgetting"},
         "ready", "swallowed exception"),
    ],
)
def test_stand_in_loop_failures_are_caught(
    tmp_path: Path, name: str, kwargs: dict, settle_kind: str, phrase: str
) -> None:
    ready = threading.Event()
    settle = _after_loop_ends if settle_kind == "dead" else _after_ready(ready)
    with pytest.raises(AssertionError, match=phrase):
        _stress_body(
            tmp_path,
            loop_target=_stand_in(ready, **kwargs),
            settle=settle,
            **_FAST,
            **_FAST_TEARDOWN,
        )


def test_stand_in_loop_that_ignores_stop_is_caught(tmp_path: Path) -> None:
    ready = threading.Event()
    release = threading.Event()
    try:
        with pytest.raises(AssertionError, match="did not join"):
            _stress_body(
                tmp_path,
                loop_target=_stand_in(ready, release=release, ignores_stop=True),
                settle=_after_ready(ready),
                **_FAST,
                **_FAST_TEARDOWN,
            )
    finally:
        release.set()


@pytest.mark.parametrize(
    ("writer", "writer_is_dead", "phrase"),
    [
        (_returning_writer, True, "writer thread was not alive"),
        (_crashing_writer, True, "writer thread raised"),
        (_idle_writer, False, "no successful bump_recall"),
    ],
)
def test_stand_in_writer_failures_are_caught(
    tmp_path: Path, writer, writer_is_dead: bool, phrase: str  # noqa: ANN001
) -> None:
    ready = threading.Event()
    with pytest.raises(AssertionError, match=phrase):
        _stress_body(
            tmp_path,
            loop_target=_stand_in(ready),
            writer_target=writer,
            settle=(
                _after_writer_ends(ready)
                if writer_is_dead
                else _after_ready(ready, wait_for_bump=False)
            ),
            **_FAST,
            **_FAST_TEARDOWN,
        )


def test_stand_in_writer_that_ignores_stop_is_caught(tmp_path: Path) -> None:
    ready = threading.Event()
    release = threading.Event()
    try:
        with pytest.raises(AssertionError, match="writer thread did not join"):
            _stress_body(
                tmp_path,
                loop_target=_stand_in(ready),
                writer_target=_writer_ignoring_stop(release),
                settle=_after_ready(ready),
                **_FAST,
                **_FAST_TEARDOWN,
            )
    finally:
        release.set()


def test_swallowed_error_capture_ignores_debug_records_and_other_loggers() -> None:
    """The handler must not turn a lowered root log level, or an expected
    exception from a logger the loop does not own, into a hit."""
    capture = _SwallowedErrorCapture()
    try:
        raise ValueError("x")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    def _record(name: str, level: int, info) -> logging.LogRecord:  # noqa: ANN001
        return logging.LogRecord(name, level, __file__, 1, "msg", None, info)

    capture.emit(_record("brain.bridge.supervisor", logging.DEBUG, exc_info))
    capture.emit(_record("brain.memory.floor_calibration", logging.ERROR, exc_info))
    capture.emit(_record("brain.bridge.supervisor", logging.ERROR, None))
    assert capture.hits == []
    capture.emit(_record("brain.bridge.supervisor", logging.ERROR, exc_info))
    capture.emit(_record("brain.forgetting", logging.WARNING, exc_info))
    assert len(capture.hits) == 2
