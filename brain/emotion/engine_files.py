"""The files the emotion engine owns, and the one place that knows their paths and their lock (Engine 0, #575; Engine 2, #577).

Everything lives under ``<persona_dir>/state/`` (``paths.state_dir``); no code of ours creates that folder, because
``write_json``, ``write_jsonl``, ``append_jsonl`` and ``file_lock`` each create a missing parent.

* ``emotion_state.json`` (+ ``.bak1``-``.bak3``): the averages, the weights and a 30-day snapshot per emotion, and the
  persona's one switch-over time, ``switch_over_at`` (the only record that the persona was switched over; there is no separate
  switch file: ``switch_over_state`` reads "switched" off a loaded state), and ``rows_through``, the highest ``memories``
  rowid the engine has seen (``EngineState.rows_through``; null when none). Written with
  ``write_json(backups=3, indent=None)``. Read through ``json_io.read_json_walk``, the read-only walk with the validator
  below: a read never heals, quarantines, rotates or rewrites anything, and says which copy it loaded. ``read_state`` reads
  fresh objects (what a drain mutates); ``StateCache`` keeps the parsed state for readers by a stat key.
* ``emotion_pending.jsonl``: the queue of events waiting to be folded, one JSON event per line. Append-only: an append writes
  only its own line, so it costs the same at any depth (a whole-file rewrite per append was quadratic). No backups (a drained
  queue must not come back from one). It is read by position: ``read_queue(persona_dir, from_line=N)`` returns the lines after
  line N and the line count, and the position is kept in the state (``EngineState.queue_line``). The drain empties the file
  after it has folded the events (``empty_queue``); the engine never deletes it. A missing file is an empty queue; a bad line
  is dropped on its own with an error log line (``read_queue``); a last line with no newline yet is not a line and is read
  whole once it is complete.
* ``emotion_changes/YYYY-MM-DD.jsonl``: the change log, one file a day, a crash-recovery record of chat-derived changes.
  Whole files are deleted once they are older than the retention boundary.
* ``emotion_vocabulary.json`` (the persona's own file, written only through ``brain.emotion.vocabulary_file``): not touched here;
  the half-lives of her entries come from ``brain.emotion.vocabulary_access.lookup`` (``VocabularyHalfLives``, behind
  ``EmotionEngine.half_life``).
* ``<persona_dir>/engine_progress.json`` (NOT under ``state/``, and outside the engine lock): the progress record the x10 fix
  and the first rebuild write while they run, so the app can show the wait (``EngineProgress`` writes it, ``read_engine_progress``
  returns it while its writer is alive; see the section after ``state_lock``).
* ``<persona_dir>/rebuild_notice.json`` (also outside the lock): why a switch-over did not complete, while the persona is not
  switched over, so the window can say so (``settle_rebuild_notice`` writes and clears it, ``read_rebuild_notice`` reads it; see
  the section after ``EngineProgress``). Unlike the progress record it is not a live writer's: it outlives the process that wrote it.
* ``<persona_dir>/command_runs/<pid>.json`` (also outside the lock): the run record of each command process that touches the
  engine, so a start that rebuilds can find and end it (``register_command_run``, ``live_command_runs``).

The lock. ONE lock guards the four files: ``state_lock(persona_dir)``, a non-re-entrant ``file_lock`` of the state file that
waits at most ``dev_constants.EMOTION_LOCK_WAIT_S`` (``file_lock.locked_within``) and raises ``LockTimeoutError`` when the wait
runs out. Every acquisition of it passes the wait; what a caller does on the timeout is that caller's. No other function in
this module takes it; every caller holds it (a hook around its appends, a drain around its whole run, the rebuild around
everything) and no code path may take it twice. Lock-free by design: ``read_state``, ``StateCache``, ``switch_over_state``,
``read_queue``, ``read_changes`` and ``list_change_files`` (a reader never waits for a writer; the state file is replaced
atomically, the queue is appended to and emptied atomically, the change log is appended to and
deleted whole; a queue reader that is not the drain, which holds the lock, can meet a line being appended and does not count
it until it is complete). Expecting the caller's lock: ``save_state``, ``append_event``, ``empty_queue``, ``append_change``
and ``prune_changes``.

Times are UTC ISO strings written with ``iso_utc`` (``Z``) and read with ``lenient_utc``; day labels are local ``YYYY-MM-DD``
from ``engine_utils.day_of``. All constants (file and folder names, the retention length, the value cap) are read as
``dev_constants.NAME`` at call time.

The half-life lookup (Engine 0, #801 section 5). ``VocabularyHalfLives`` asks ``vocabulary_access.lookup`` for the half-life of one
of her entries; it reads no file itself. A value that is not a finite number above 0 is reported (throttled) and the built-in
half-life applies.
"""

from __future__ import annotations

import atexit
import logging
import math
import reprlib
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from brain import dev_constants, paths, process_identity
from brain.emotion import vocabulary, vocabulary_access
from brain.emotion.engine_utils import (
    _MODIFIER_FLOOR,  # with _MODIFIER_SPAN: the range `source_modifier` makes, which a stored modifier is held to
    _MODIFIER_SPAN,
    Average,
    EmotionState,
    EngineState,
    Event,
    RepeatEntry,
    RunRecord,
    SnapshotDay,
    _hold,  # the one hold-to-0-10 rule, shared with the fold (same package, no second copy)
    day_of,
)
from brain.health.json_io import (
    Recovery,
    append_jsonl,
    iter_jsonl_raw,
    read_json,
    read_json_walk,
    read_jsonl_from,
    write_json,
    write_jsonl,
)
from brain.utils import file_lock, file_ops
from brain.utils.log_throttle import count_throttled, warn_throttled
from brain.utils.time import iso_utc, lenient_utc

logger = logging.getLogger(__name__)

# The state file keeps three backups; `json_io.read_json_walk` reads exactly `.bak1` to `.bak3`.
_BACKUPS = 3

class VocabularyHalfLives:
    """The numeric half-lives her vocabulary gives, by canonical name, through `vocabulary_access.lookup` (#801 section 5).

    In the bridge the lookup is the registry; any other process asks the bridge (and, with no bridge, reads her file read-only for
    the run). A framework (core, complex, body) name is answered by the caller's built-in table, never from here, so it costs no
    call. The caller asks for a whole batch (one memory's names) in one lookup, never one call per name (spec section 5). Nothing is
    kept between calls: a process holds no copy of her vocabulary, so a change made in the bridge is seen by the next call.
    """

    def __init__(self, persona_dir: Path) -> None:
        self._persona_dir = Path(persona_dir)

    def get_many(self, names: Iterable[str]) -> dict[str, float]:
        """Her vocabulary's numeric half-life for each of `names` that has one, keyed by the tidied name: ONE lookup for the batch.

        A name missing from the answer has no usable number in her vocabulary (a framework name, not listed, a null entry
        "never fades", a value that is not a finite number above 0, or the lookup failed): the caller goes on to the built-in
        table. A framework name costs nothing (it is never asked), and a batch of only framework names makes no call.
        """
        wanted = sorted({vocabulary.canonical_name(n) for n in names if not vocabulary.is_baseline(n)})
        if not wanted:
            return {}
        out: dict[str, float] = {}
        for name, found in vocabulary_access.lookup(self._persona_dir, wanted).items():
            days = None if found is None else self._usable(name, found.half_life_days)
            if days is not None:
                out[name] = days
        return out

    @staticmethod
    def _usable(name: str, days: float | None) -> float | None:
        if days is None:
            return None
        if not math.isfinite(days) or days <= 0:
            warn_throttled(
                logger,
                f"engine_files:half_life:{name}",
                "emotion vocabulary: the half-life of %r must be a finite number above 0, found %r; the built-in half-life applies to it",
                name,
                days,
            )
            return None
        return days


# ================================================================================================
# Paths and the lock
# ================================================================================================


def state_path(persona_dir: Path) -> Path:
    return paths.state_dir(persona_dir) / dev_constants.EMOTION_STATE_FILENAME


def queue_path(persona_dir: Path) -> Path:
    return paths.state_dir(persona_dir) / dev_constants.EMOTION_PENDING_FILENAME


def changes_dir(persona_dir: Path) -> Path:
    return paths.state_dir(persona_dir) / dev_constants.EMOTION_CHANGES_DIRNAME


def state_lock(persona_dir: Path) -> AbstractContextManager[None]:
    """The one lock of the engine's four files: the lock of the state file (sidecar ``emotion_state.json.lock``), bounded.

    ``file_lock.locked_within`` of the state file path (the lock adds the sidecar name itself), waiting at most
    ``dev_constants.EMOTION_LOCK_WAIT_S``, read here at call time. When the wait runs out it raises ``LockTimeoutError`` and
    the ``with`` body never runs. The lock is NOT re-entrant (taking it again inside the body times out). The hooks, the drain
    and the rebuild take it through this helper, once each; nothing else in this module takes it (see the module docstring).

    ``LockTimeoutError`` is an ``OSError`` (a ``TimeoutError``): a caller that wraps the ``with`` in ``except OSError`` would
    swallow it, so it catches ``LockTimeoutError`` first. What a caller does on the timeout is that caller's.
    """
    return file_lock.locked_within(state_path(persona_dir), dev_constants.EMOTION_LOCK_WAIT_S)


# ================================================================================================
# The progress record of a long start-up step (HR5 of #633; owner S101 and S102)
# ================================================================================================
#
# ``engine_progress.json`` sits in the PERSONA directory beside ``bridge.json`` and the bridge's watcher files (not under
# ``state/``), and is written outside the engine lock's protection: it is a display record, not engine state. Two steps write it,
# each only when it is running: the x10 fix ("rescaling", no count) and the first rebuild ("rebuilding", with a count). The
# reader returns it only while its writer is alive (``process_identity.holder_alive``): no age test, no heartbeat, so a live writer
# counts however long a phase between writes takes. It calls ``json_io`` directly (``brain.emotion`` does not import
# ``brain.bridge``). It is a LIVE writer's record and says nothing about a step that failed: a switch-over that did not complete
# is remembered by the rebuild notice (the section after ``EngineProgress``), which outlives its writer.

_PROGRESS_FORMAT_VERSION = 1


def progress_path(persona_dir: Path) -> Path:
    return Path(persona_dir) / dev_constants.ENGINE_PROGRESS_FILENAME


def progress_record(
    *, pid: int, pid_token: str | None, persona: str, phase: str, done: int | None, total: int | None, updated_at: str
) -> dict[str, Any]:
    """The record's one shape (golden fixture: tests/fixtures/bridge_watcher_files/engine_progress.json).

    ``phase`` "rescaling" (the x10 fix, no count) or "rebuilding" (the first rebuild, and the rebuild of a roll-forward at start-up); ``done`` and ``total`` are None until
    the rebuild knows its window (shown as "preparing") and for the x10 step.
    """
    return {
        "version": _PROGRESS_FORMAT_VERSION,
        "pid": pid,
        "pid_token": pid_token,
        "persona": persona,
        "phase": phase,
        "done": done,
        "total": total,
        "updated_at": updated_at,
    }


def read_progress(persona_dir: Path) -> dict[str, Any] | None:
    """The record as written, or None when missing or unreadable as a JSON object (like ``shutdown_progress.read_record``)."""
    try:
        return read_json(progress_path(persona_dir), default=lambda: None, recover=Recovery.CONTENT, expect=dict)
    except Exception:  # noqa: BLE001 - an unreadable display record is "no record"
        return None


def _pid_of(record: dict[str, Any] | None) -> int | None:
    """The record's pid when it is a real process number (an int, not a bool, that the platform can hold as a pid), else None.

    A number beyond the platform's range names no process and makes the OS pid check raise; it is "no usable pid" here, so the
    one liveness test below never has to read that failure as alive (B59 HQD bycatch).
    """
    pid = record.get("pid") if isinstance(record, dict) else None
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and process_identity.is_pid_number(pid) else None


def _may_be_alive(pid: int, token: str | None, path: Path) -> bool:
    """The one liveness test for the records of this module (progress records and command run records): is the writer of the
    record at ``path`` alive (``holder_alive``: an unknown answer is alive; no token: a pid check)? A check that itself fails
    counts as alive: a record that may name a live process is never dropped."""
    try:
        return process_identity.holder_alive(pid, token)
    except Exception as exc:  # noqa: BLE001 - undecidable means "may be alive"
        warn_throttled(
            logger,
            f"record-writer:alive:{path}",
            "emotion engine: could not tell whether the writer of the record %s is alive (%s: %s); read as alive",
            path,
            type(exc).__name__,
            exc,
            exc_info=False,
        )
        return True


def _writer_alive(record: dict[str, Any], path: Path) -> bool:
    """Is the process that wrote the progress ``record`` (read from ``path``) alive? No usable pid: not alive."""
    pid = _pid_of(record)
    token = record.get("pid_token")
    return pid is not None and _may_be_alive(pid, token if isinstance(token, str) else None, path)


def read_engine_progress(persona_dir: Path) -> dict[str, Any] | None:
    """The progress record, only while its writer is alive; None for no record, a record that is not a mapping or has no usable
    pid, and a writer that is gone (a record a crashed or killed process left behind). Feeds ``daemon.status_dict``."""
    try:
        record = read_progress(persona_dir)
        return record if record is not None and _writer_alive(record, progress_path(persona_dir)) else None
    except Exception:  # noqa: BLE001 - a status read must not fail on a display record
        return None


class EngineProgress:
    """One start-up step's writer of the progress record. Never raises: a failed write or removal is logged and the step goes on.

    ``write()`` at the start of the step (no count), ``report(done, total)`` as it goes (written only when the whole-number
    percentage changed, so at most 101 counted writes), ``remove()`` when the step ends, success or failure (the caller's
    ``finally``). Every write first reads the record there: one that names ANOTHER live process (the x10 step of a second
    start) is not replaced, and the next write goes through once that record is gone (B47b HR5b (d)). A percentage counts as
    tried when it is attempted, so a blocked writer does not read the file again for every row. ``remove()`` deletes only a record
    that names this process's pid and token (``process_identity.is_own_identity``); with no token to read (``own_token()`` is
    None) the record carries none and is still removed, so a bridge that cannot read its own token does not leave a record
    that reads as live for as long as it runs. A record that names a token is not removed by a writer with none.
    """

    def __init__(self, persona_dir: Path, phase: str) -> None:
        self._persona_dir = Path(persona_dir)
        self._phase = phase
        self._last_percent: int | None = None

    def write(self) -> None:
        """The entry write: this phase, no count."""
        self._safely("write", self._put, None, None)

    def report(self, done: int, total: int) -> None:
        self._safely("write", self._report, done, total)

    def remove(self) -> None:
        self._safely("remove", self._remove)

    def _safely(self, what: str, action: Callable[..., None], *args: int | None) -> None:
        try:
            action(*args)
        except Exception as exc:  # noqa: BLE001 - a display record must never fail the step it describes
            warn_throttled(
                logger,
                f"engine-progress:{what}:{self._persona_dir}",
                "emotion engine: could not %s the progress record in %s (%s: %s); start-up goes on",
                what,
                self._persona_dir,
                type(exc).__name__,
                exc,
                exc_info=False,
            )

    def _report(self, done: int, total: int) -> None:
        if total <= 0:
            return  # an empty window shows no count
        percent = max(0, min(100, done * 100 // total))
        if percent == self._last_percent:
            return
        self._last_percent = percent
        self._put(done, total)

    def _put(self, done: int | None, total: int | None) -> None:
        existing = read_progress(self._persona_dir)
        if existing is not None and _pid_of(existing) != process_identity.own_pid() and _writer_alive(existing, progress_path(self._persona_dir)):
            return  # another live process's record (a second start's x10 step): never replaced
        record = progress_record(
            pid=process_identity.own_pid(),
            pid_token=process_identity.own_token(),
            persona=self._persona_dir.name,
            phase=self._phase,
            done=done,
            total=total,
            updated_at=iso_utc(datetime.now(UTC)),
        )
        write_json(progress_path(self._persona_dir), record, indent=2, sort_keys=True, fsync=False, mode=0o600)

    def _remove(self) -> None:
        record = read_progress(self._persona_dir)
        if isinstance(record, dict) and process_identity.is_own_identity(
            record.get("pid"), record.get("pid_token")
        ):
            file_ops.delete_file(progress_path(self._persona_dir))


# --- the rebuild notice: a switch-over that did not complete (Hana's 11-18 review of #633, point 5; owner S112) ---------------------
#
# ``rebuild_notice.json`` in the PERSONA directory holds why the persona's switch-over did not complete and the bridge started anyway
# on the state saved before: ``{"pid", "pid_token", "reason": <one line>}`` (the writer's identity in the progress record's shape, so
# a start can tell the notice its own rebuild child wrote from another; no time is recorded). It exists exactly while, after an attempt, the persona is NOT
# switched over (``switch_over_state(read_state(...)) is True`` is the test; an unreadable state is not switched), so a switched
# persona whose roll-forward fails never gets one (that failure stays in the runtime log). ``build_persona_state`` reads it into the
# window's state; the full error is in the runtime log. ONE writer, ``settle_rebuild_notice``, is called by the engine (the failure
# points inside the rebuild process) and by the start process (``daemon``: a rebuild child that could not be started or ended
# non-zero, a need check that failed), and clears it: the rebuild child when its ready step returns (``cli`` handler), the start
# process when its need check answers no. Not reused: ``shutdown_progress.write_record/read_record`` live in ``brain.bridge``, which the
# engine may not import, and a daemon-only use would make two writers; ``EngineProgress`` is for a LIVE writer (it refuses another live
# process's record, removes by pid, is read only while the writer lives), and a failure record must outlive its writer. Also not
# chosen: a flag derived from ``is_live`` the way ``recovering`` is derived (no file), which cannot say what happened (S112: "saying
# what happened"; P9b), and the start's stderr line, which the app drops for a start that exits 0. No lock: the
# write replaces the file atomically, and each writer states a real failure of a not-switched persona, so the last stands.


def rebuild_notice_path(persona_dir: Path) -> Path:
    return Path(persona_dir) / dev_constants.REBUILD_NOTICE_FILENAME


def rebuild_notice_record(*, pid: int, pid_token: str | None, reason: str) -> dict[str, Any]:
    """The record's one shape (golden fixture: tests/fixtures/bridge_watcher_files/rebuild_notice.json).

    ``pid`` and ``pid_token`` name the process that wrote it, in the shape of the progress record, so the start process can ask
    ``process_identity.names_this_bridge`` whether the notice was written by the rebuild child it launched (no clock is read for that).
    """
    return {"pid": pid, "pid_token": pid_token, "reason": reason}


def read_rebuild_notice(persona_dir: Path) -> dict[str, Any] | None:
    """The notice as written, or None when missing, unreadable as a JSON object, or without a text ``reason``. Never raises."""
    try:
        record = read_json(rebuild_notice_path(persona_dir), default=lambda: None, recover=Recovery.CONTENT, expect=dict)
    except Exception:  # noqa: BLE001 - an unreadable display record is "no notice"
        return None
    reason = record.get("reason") if isinstance(record, dict) else None
    return record if isinstance(reason, str) and reason else None


def settle_rebuild_notice(persona_dir: Path, reason: str | None, *, keep_notice_of: tuple[int, str] | None = None) -> None:
    """Record, keep or clear the rebuild notice after an attempt. The one writer; never raises.

    ``reason`` is why the switch-over did not complete (made one line here), or None for an attempt that completed or a check that
    found nothing to do. ``keep_notice_of`` is the ``(pid, token)`` of the rebuild child a start process launched. In order:
    1. No notice file and ``reason`` None: return (the healthy path costs one stat; a stat that raises is logged and read as "a
       notice exists", and a notice that cannot be read still counts as existing).
    2. The persona is switched over: delete the notice and return (a switched persona never has one).
    3. ``reason`` None: return (the persona is not switched, so what the notice says still holds).
    4. ``keep_notice_of`` given and the notice names exactly that pid and token (``process_identity.names_this_bridge``): return,
       the launched child's own, more exact notice stands. Decided by process identity, not by clock times (a clock stepped back
       while the child ran cannot replace its reason). A notice that cannot be read, or has no identity (a record from before the
       identity was recorded), or a malformed one, or another process's, is not the child's and is replaced.
    5. Write the notice with this process's own pid and token, only when the persona folder exists (the writer never creates one).

    A failure of any step is logged by ``warn_throttled`` and the function returns. Accepted edge (P4, HRF red-team G2): a delete that
    fails (step 2, after the file layer's own retry) leaves a notice on a persona that is now switched over; the window then shows
    "not brought up to date" until the next start's or command's need check clears it (the reader is a pure read and does not test
    the state), and the failure is logged. Accepted edge (P4, Hana's point 5): a notice that
    cannot be written leaves the window showing nothing; the original error is in the runtime log. The notice sits in the persona
    folder where the bridge's own state file is written, so an unwritable folder mostly stops the bridge too (and the error screen
    shows); the invisible case needs an incomplete switch-over and only this small write failing, read as well under 1 in 10,000
    starts (not measured).
    """
    stage = "check"
    try:
        path = rebuild_notice_path(persona_dir)
        try:
            exists = file_ops.stat_path(path) is not None
        except Exception as exc:  # noqa: BLE001 - read as "a notice exists": the next steps decide
            _notice_failed(persona_dir, "check", exc)
            exists = True
        if not exists and reason is None:
            return
        if switch_over_state(read_state(persona_dir)) is True:
            stage = "delete"
            file_ops.delete_file(path)
            return
        if reason is None:
            return
        stage = "read"
        if keep_notice_of is not None:
            child_pid, child_token = keep_notice_of
            if process_identity.names_this_bridge(read_rebuild_notice(persona_dir), child_pid, child_token):
                return
        stage = "write"
        folder = file_ops.stat_path(persona_dir)
        if folder is None or not folder.is_dir:
            return
        record = rebuild_notice_record(
            pid=process_identity.own_pid(), pid_token=process_identity.own_token(), reason=" ".join(str(reason).split())
        )
        write_json(path, record, indent=2, sort_keys=True, fsync=False, mode=0o600)
    except Exception as exc:  # noqa: BLE001 - a display record must never fail the step it describes
        _notice_failed(persona_dir, stage, exc)


def _notice_failed(persona_dir: Path, stage: str, exc: BaseException) -> None:
    warn_throttled(
        logger,
        f"rebuild-notice:{stage}:{persona_dir}",
        "emotion engine: could not %s the rebuild notice in %s (%s: %s); the window will not show it",
        stage,
        persona_dir,
        type(exc).__name__,
        exc,
        exc_info=False,
    )


# --- the run records of commands (HS3 of #633) ----------------------------------------------------------------------------------
#
# One record per command process that touches the engine, ``<persona_dir>/command_runs/<pid>.json``, written when the command
# starts (``register_command_run``) and removed when it exits. A start that must rebuild reads them (``live_command_runs``) and
# ends each command named there (``kill.kill_command``), because no other mechanism finds a running command. Like the progress
# record these are outside the engine lock and are not memory: a failed write or delete is logged and the step goes on. A record that
# cannot be read is never dropped while the pid in its file name may be alive: the reader returns it marked ``UNREADABLE_RUN_KEY``
# and the start fails naming its path (B58 HQB Q1).

_COMMAND_RUN_FORMAT_VERSION = 1

# The key of an entry ``live_command_runs`` returns for a record it cannot read whose file-name pid may be alive; the entry has
# the keys ``pid`` (from the file name) and ``path`` and no token.
UNREADABLE_RUN_KEY = "unreadable"


def command_runs_dir(persona_dir: Path) -> Path:
    return Path(persona_dir) / dev_constants.COMMAND_RUNS_DIRNAME


def command_run_path(persona_dir: Path, pid: int) -> Path:
    return command_runs_dir(persona_dir) / f"{pid}.json"


def command_run_record(*, pid: int, pid_token: str | None, persona: str) -> dict[str, Any]:
    """The run record's one shape (golden fixture: tests/fixtures/bridge_watcher_files/command_run.json)."""
    return {"version": _COMMAND_RUN_FORMAT_VERSION, "pid": pid, "pid_token": pid_token, "persona": persona}


def register_command_run(persona_dir: Path) -> None:
    """Write this process's run record and remove it at exit (``atexit``), only if it still names this process.

    A failed write is logged once and the command goes on unregistered: a start cannot then find it. A process that is killed
    leaves its record; the next ``live_command_runs`` finds the writer dead and deletes it. ``atexit`` does not run on SIGKILL,
    and not on a signal that ends the process without Python's own exit (SIGTERM with no handler).
    """
    persona_dir = Path(persona_dir)
    record = command_run_record(
        pid=process_identity.own_pid(), pid_token=process_identity.own_token(), persona=persona_dir.name
    )
    try:
        write_json(command_run_path(persona_dir, record["pid"]), record, indent=2, sort_keys=True, fsync=False, mode=0o600)
    except Exception as exc:  # noqa: BLE001 - a run record must never fail the command it describes
        warn_throttled(
            logger,
            f"command-run:write:{persona_dir}",
            "emotion engine: could not write this command's run record in %s (%s: %s); the command goes on unregistered",
            persona_dir,
            type(exc).__name__,
            exc,
            exc_info=False,
        )
        return
    atexit.register(_remove_own_command_run, persona_dir)


def _read_command_run(path: Path) -> dict[str, Any] | None:
    """The record as written; None when its content is not a JSON object (or the file is gone). An I/O fault raises."""
    return read_json(path, default=lambda: None, recover=Recovery.CONTENT, expect=dict)


def _remove_own_command_run(persona_dir: Path) -> None:
    """The ``atexit`` step: delete this process's record, and only a record that names this process (pid and token)."""
    path = command_run_path(persona_dir, process_identity.own_pid())
    try:
        record = _read_command_run(path)
        if isinstance(record, dict) and process_identity.is_own_identity(record.get("pid"), record.get("pid_token")):
            file_ops.delete_file(path)
    except Exception as exc:  # noqa: BLE001 - nothing may raise at interpreter exit
        warn_throttled(
            logger,
            f"command-run:remove:{persona_dir}",
            "emotion engine: could not remove this command's run record in %s (%s: %s)",
            persona_dir,
            type(exc).__name__,
            exc,
            exc_info=False,
        )


def _run_record_pid(name: str) -> int | None:
    """The pid a record's file name gives: ``<pid>.json`` with ASCII digits that parse as a number the platform can hold as a pid,
    else None (not a record: a name with more digits than Python converts or a number beyond the pid range is ignored, B58 HQB)."""
    stem, dot, suffix = name.rpartition(".")
    if dot != "." or suffix != "json" or not stem.isascii() or not stem.isdigit():
        return None
    try:
        pid = int(stem)
    except ValueError:  # more digits than Python converts: not a pid
        return None
    return pid if process_identity.is_pid_number(pid) else None


def _unreadable_run(pid: int, path: Path) -> dict[str, Any]:
    """The entry ``live_command_runs`` returns for a record it cannot read whose file-name pid is (or may be) alive."""
    return {UNREADABLE_RUN_KEY: True, "pid": pid, "path": str(path)}


def _delete_dead_run(path: Path) -> None:
    """Delete the record of a dead writer; a delete that fails is logged and the record skipped (it names nothing alive)."""
    try:
        file_ops.delete_file(path)
    except Exception as exc:  # noqa: BLE001 - one record that cannot be deleted must not stop the others
        warn_throttled(
            logger,
            f"command-run:delete:{path}",
            "emotion engine: could not delete the command run record %s of a command that has ended (%s: %s); skipped",
            path,
            type(exc).__name__,
            exc,
            exc_info=False,
        )


def live_command_runs(persona_dir: Path) -> list[dict[str, Any]]:
    """The run records of the other commands that may be alive, in file-name order.

    Only names of the form ``<pid>.json`` are records. A record named for this process's own pid is left out by its name, before
    it is read, so a command never refuses against its own record. A record that reads as a JSON object with a usable pid is
    decided by that pid: a dead writer (``holder_alive``) has the record deleted, a live one is returned as written (this process's
    own identity is left out here too). Any other record, one that cannot be read (an I/O fault, a folder with the name), is not a JSON
    object, or has no usable pid, is decided by the pid in its FILE NAME: dead, it is deleted; alive, or undecidable, it is kept
    and returned as ``{UNREADABLE_RUN_KEY: True, "pid", "path"}``, which the caller treats as a command that could not be ended. A
    read that answers "not a JSON object" for a file that is gone by the time it is checked (``file_ops.stat_path``: its own command
    removed it at exit) is skipped. A delete that fails is logged and that record skipped.

    Raises ``OSError`` when the folder cannot be listed (a missing folder is no records): the folder may hide a live command, so
    the caller fails the step naming it. Nothing else raises.

    Accepted edge (P4, B58 HQB Q1, B60 HQT): an unreadable record, or a readable one with no token (a command that could not read
    its own identity), whose pid was reused by an unrelated live process blocks starts until that process exits or the user removes
    the file the caller's line names.

    Accepted edge (P4, B58 HQB Q2): a person runs one of the six commands by hand inside a shell spawned from another registered
    command. The step's ``kill.kill_command`` then suspends every descendant of the outer command, the inner command included, so
    the inner command freezes before it can end anything and both stay stopped until ended by hand. Nothing is lost beyond the
    in-flight work. No code path of the product starts one of the six from inside another. A walk of the caller's ancestors to
    leave them out is rejected (P17): new code for a case the product does not create.
    """
    persona_dir = Path(persona_dir)
    folder = command_runs_dir(persona_dir)
    names = file_ops.list_dir(folder)
    own_pid = process_identity.own_pid()
    runs: list[dict[str, Any]] = []
    for name in names:
        name_pid = _run_record_pid(name)
        if name_pid is None or name_pid == own_pid:
            continue
        path = folder / name
        try:
            record = _read_command_run(path)
        except Exception as exc:  # noqa: BLE001 - an I/O fault: the record may name a live command, so the name pid decides
            warn_throttled(
                logger,
                f"command-run:read:{path}",
                "emotion engine: could not read the command run record %s (%s: %s); decided by the pid in its name",
                path,
                type(exc).__name__,
                exc,
                exc_info=False,
            )
            record = None
        else:
            if record is None:
                try:
                    if file_ops.stat_path(path) is None:
                        continue  # gone: its own command removed it at exit
                except Exception as exc:  # noqa: BLE001 - cannot tell if it is gone: the name pid decides
                    warn_throttled(
                        logger,
                        f"command-run:stat:{path}",
                        "emotion engine: could not look at the command run record %s (%s: %s); decided by the pid in its name",
                        path,
                        type(exc).__name__,
                        exc,
                        exc_info=False,
                    )
        record_pid = _pid_of(record)
        # The file name is the record's identity (B58 HQB Q1): a record whose body names another pid than its file is not a valid
        # record and is decided by the name pid below, so every record returned as read has the pid of its file and a caller
        # building its path from the pid (the not-ended line, B60 HQT) names the right file.
        if record is not None and record_pid is not None and record_pid == name_pid:
            token = record.get("pid_token")
            if not _may_be_alive(record_pid, token if isinstance(token, str) else None, path):
                _delete_dead_run(path)
            elif not process_identity.is_own_identity(record_pid, token):
                runs.append(record)
            continue
        if _may_be_alive(name_pid, None, path):
            runs.append(_unreadable_run(name_pid, path))
        else:
            _delete_dead_run(path)
    return runs


def peer_trust(persona_dir: Path, peer_id: str) -> float:
    """How far this persona trusts one peer (0 to 1): the engine's one door to the peer-trust score.

    Passes straight through to the kindled-link accessor ``brain.kindled_link.relationship.peer_trust``, which answers 0.0 for
    no database, no row, an error or an unreadable row and creates nothing. The engine calls it only through here (tests wrap
    and patch this name). The import is inside the function because that module imports the memory store.
    """
    from brain.kindled_link.relationship import peer_trust as _peer_trust

    return _peer_trust(persona_dir, peer_id)


# ================================================================================================
# The state file
# ================================================================================================


def _finite(value: object, what: str) -> float:
    """A finite number (not a bool, not too large for a float), else ValueError naming `what`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, found {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{what} is too large to be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, found {number!r}")
    return number


def _time(value: object, what: str) -> datetime:
    """A time the engine can use: text `lenient_utc` accepts and `day_of` can turn into a local day, else ValueError.

    The second test is not decoration: a time that parses but cannot be converted (year 1, or before 1970 on Windows) would
    make every close of that state fail for ever.
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a time as text, found {type(value).__name__}")
    parsed = lenient_utc(value)
    if parsed is None:
        raise ValueError(f"{what} is not a time: {value!r}")
    try:
        day_of(parsed)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{what} cannot be turned into a local day: {value!r}") from exc
    return parsed


def _day(value: object, what: str) -> str:
    """A real calendar day written exactly ``YYYY-MM-DD``, else ValueError ("20260101" parses but is not the form)."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a day as text, found {type(value).__name__}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{what} is not a day: {value!r}") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{what} is not written YYYY-MM-DD: {value!r}")
    return value


def _mapping(parent: Mapping, key: str, what: str) -> Mapping:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{what}.{key} must be an object, found {type(value).__name__ if key in parent else 'nothing'}")
    return value


def validate_state(data: object) -> None:
    """Raise ValueError unless `data` is a state the engine can use (the validator `read_json_walk` applies to every copy).

    Rejects: not an object; a missing or unparseable `saved_at`; an unparseable `reset_at`, `last_update`, `switch_over_at`
    or `persona_start_at` (a missing or null `persona_start_at` is accepted: a closed gate until the first memory; a missing or
    null `switch_over_at` is accepted: the persona is not switched over); an
    `active_day` that is not a real day, or unset while `last_update` is set; `emotions` or `repeats` of the wrong type; an
    emotion that is not an object or lacks `short`, `long` or `last_entry_day`; a non-finite number, a bool, or a negative
    weight in an average; a `last_entry_day` that is not a real day or is later than `active_day`. An average outside 0-10 is
    NOT a rejection (`state_from_json` holds it), and neither is anything wrong with a snapshot or a repeat record (those are
    dropped one at a time, never the whole state), a `queue_line`, or a `rows_through` (both read as their default). Every
    refusal is a ValueError.
    """
    if not isinstance(data, dict):
        raise ValueError(f"the state must be an object, found {type(data).__name__}")
    if "saved_at" not in data:
        raise ValueError("saved_at is missing")
    _time(data["saved_at"], "saved_at")
    # Accepted edge (Planning B28 SW3, P4, falls under the open owner question Q17): a malformed `switch_over_at` makes the
    # WHOLE state unreadable here, exactly like a malformed `persona_start_at` (one validation rule for the state, P1), so
    # the walk falls to a backup copy, and when no copy is readable the persona reads as unreadable (not switched, not cached).
    for key in ("reset_at", "last_update", "persona_start_at", "switch_over_at"):
        if data.get(key) is not None:
            _time(data[key], key)
    active_day = data.get("active_day")
    if active_day is not None:
        _day(active_day, "active_day")
    elif data.get("last_update") is not None:
        raise ValueError("active_day is unset while last_update is set")
    if "repeats" in data and not isinstance(data["repeats"], dict):
        raise ValueError(f"repeats must be an object, found {type(data['repeats']).__name__}")
    emotions = data.get("emotions")
    if not isinstance(emotions, dict):
        raise ValueError(f"emotions must be an object, found {type(emotions).__name__ if 'emotions' in data else 'nothing'}")
    for name, emotion in emotions.items():
        where = f"emotion {name!r}"
        if not isinstance(emotion, dict):
            raise ValueError(f"{where} must be an object, found {type(emotion).__name__}")
        for key in ("short", "long"):
            average = _mapping(emotion, key, where)
            for field in ("avg", "weight"):
                if field not in average:
                    raise ValueError(f"{where}.{key}.{field} is missing")
                number = _finite(average[field], f"{where}.{key}.{field}")
                if field == "weight" and number < 0:
                    raise ValueError(f"{where}.{key}.weight must not be negative, found {number!r}")
        if "last_entry_day" not in emotion:
            raise ValueError(f"{where}.last_entry_day is missing")
        last_entry_day = emotion["last_entry_day"]
        if last_entry_day is not None:
            _day(last_entry_day, f"{where}.last_entry_day")
            if active_day is not None and last_entry_day > active_day:
                raise ValueError(f"{where}.last_entry_day {last_entry_day} is later than active_day {active_day}")


def _snapshot_day(item: object, previous_day: str | None) -> SnapshotDay:
    """One snapshot day, or ValueError saying what is wrong with it. Averages outside 0-10 are held, not refused.

    Only the known leading fields are read: a day with more than five fields, or a run record with more than three, is read
    by its first five (three), so a state written by a newer version is not dropped after a rollback (additive-only). Fewer is invalid.
    """
    if not isinstance(item, list) or len(item) < 5:
        raise ValueError("a snapshot day is an array of at least five: day, short, long, felt, run")
    day = _day(item[0], "day")
    if previous_day is not None and day <= previous_day:
        raise ValueError(f"day {day} is not later than the day before it, {previous_day}")
    short = _hold(_finite(item[1], "short average"))
    long = _hold(_finite(item[2], "long average"))
    felt = item[3]
    if type(felt) is not int or felt not in (0, 1):
        raise ValueError(f"felt must be 0 or 1, found {felt!r}")
    run: RunRecord | None = None
    if item[4] is not None:
        record = item[4]
        if not isinstance(record, list) or len(record) < 3:
            raise ValueError("a run record is null or an array of at least three: start day, direction, last day with the direction")
        start = _day(record[0], "run start day")
        direction = record[1]
        if type(direction) is not int or direction not in (1, -1):
            raise ValueError(f"run direction must be 1 or -1, found {direction!r}")
        last_with = _day(record[2], "run last with-direction day")
        # A start later than the day is refused by the next two checks: its last with-direction day would be after the day
        # or before the start.
        if last_with > day:
            raise ValueError(f"run last with-direction day {last_with} is later than the day {day}")
        if last_with < start:
            raise ValueError(f"run last with-direction day {last_with} is before its start day {start}")
        if (date.fromisoformat(day) - date.fromisoformat(last_with)).days > 1:
            raise ValueError(f"run last with-direction day {last_with} is more than one day before the day {day}")
        run = RunRecord(start, direction, last_with)
    return SnapshotDay(day, short, long, bool(felt), run)


def _snapshot_from_json(name: str, raw: object, label: str) -> list[SnapshotDay]:
    if not isinstance(raw, list):
        count_throttled(
            logger,
            f"engine_files:snapshot:{label}:{name}",
            logging.ERROR,
            "emotion state %s: emotion %r has no usable snapshot list; treated as an empty snapshot",
            label,
            name,
        )
        return []
    days: list[SnapshotDay] = []
    previous: str | None = None
    for index, item in enumerate(raw):
        try:
            day = _snapshot_day(item, previous)
        except ValueError as exc:
            count_throttled(
                logger,
                f"engine_files:snapshot:{label}:{name}:{index}",
                logging.ERROR,
                "emotion state %s: snapshot day %d of emotion %r dropped as a missing day (%s)",
                label,
                index,
                name,
                exc,
            )
            continue
        days.append(day)
        previous = day.day
    return days


_VALUE_REPR = reprlib.Repr()
_VALUE_REPR.maxstring = _VALUE_REPR.maxother = 40
_VALUE_REPR.maxlevel = 1
_VALUE_REPR.maxlist = _VALUE_REPR.maxdict = _VALUE_REPR.maxtuple = _VALUE_REPR.maxset = 3


def _queue_line_from(data: Mapping, label: str) -> int:
    """The stored queue position: a whole number of 0 or more, else 0 with a log line (a value outside its range is held to it; `docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.9).

    A missing key reads as 0 too (a state written before the position existed). `validate_state` never refuses a state for this
    key: the worst a damaged position does is make the next drain fold the queue from its start. The log line is throttled per
    label and kind of fault; its key never holds the value, and its text carries only a short, capped form of it.
    """
    if "queue_line" not in data:
        found, shown = "missing", "missing"
    else:
        value = data["queue_line"]
        if type(value) is int and value >= 0:
            return value
        found = type(value).__name__
        shown = _VALUE_REPR.repr(value)
    count_throttled(
        logger,
        f"engine_files:queue_line:{label}:{found}",
        logging.ERROR,
        "emotion state %s: queue_line is %s, read as 0 (it must be a whole number, 0 or more); the next drain folds the queue from its start",
        label,
        shown,
    )
    return 0


def _rows_through_from(data: Mapping, label: str) -> int | None:
    """The stored `rows_through`: a whole number of 0 or more (a bool is not one), else None (never an unreadable state).

    A missing or null key reads as None without a line (a state written before the field existed, or one with no row seen).
    Any other value is logged (throttled per label and kind of fault; the key never holds the value, the text carries a short,
    capped form of it) and read as None. `validate_state` never refuses a state for this key.
    """
    value = data.get("rows_through")
    if value is None:
        return None
    if type(value) is int and value >= 0:
        return value
    found = type(value).__name__
    count_throttled(
        logger,
        f"engine_files:rows_through:{label}:{found}",
        logging.ERROR,
        "emotion state %s: rows_through is %s, read as unset (it must be a whole number, 0 or more)",
        label,
        _VALUE_REPR.repr(value),
    )
    return None


def state_from_json(data: dict, *, label: str = "state") -> EngineState:
    """The `EngineState` of a state that passed `validate_state`.

    Averages (the state's and the snapshot's) are held to 0-10. A bad snapshot day is dropped as a missing day and a bad
    repeat record is dropped, each with an error log line naming the emotion or the memory; the state stays valid. A missing
    or unusable `queue_line` reads as 0, with a log line; a missing, null or unusable `rows_through` reads as None (a log
    line for an unusable one). `label` names the file in those lines.
    """
    repeats: dict[str, RepeatEntry] = {}
    for memory_id, record in data.get("repeats", {}).items():
        try:
            if not isinstance(record, dict):
                raise ValueError("a repeat record is an object with first_at and count")
            count = record.get("count")
            if type(count) is not int or count < 0:
                raise ValueError(f"count must be a whole number, 0 or more, found {count!r}")
            repeats[memory_id] = RepeatEntry(_time(record.get("first_at"), "first_at"), count)
        except ValueError as exc:
            logger.error("emotion state %s: repeat record for memory %r dropped (%s)", label, memory_id, exc)
    emotions: dict[str, EmotionState] = {}
    for name, emotion in data["emotions"].items():
        emotions[name] = EmotionState(
            short=Average(_hold(float(emotion["short"]["avg"])), float(emotion["short"]["weight"])),
            long=Average(_hold(float(emotion["long"]["avg"])), float(emotion["long"]["weight"])),
            last_entry_day=emotion["last_entry_day"],
            snapshot=_snapshot_from_json(name, emotion.get("snapshot"), label),
        )

    def optional(key: str) -> datetime | None:
        value = data.get(key)
        return None if value is None else lenient_utc(value)

    return EngineState(
        saved_at=lenient_utc(data["saved_at"]),
        reset_at=optional("reset_at"),
        last_update=optional("last_update"),
        active_day=data.get("active_day"),
        persona_start_at=optional("persona_start_at"),
        repeats=repeats,
        emotions=emotions,
        queue_line=_queue_line_from(data, label),
        switch_over_at=optional("switch_over_at"),
        rows_through=_rows_through_from(data, label),
    )


def _iso(value: datetime, what: str) -> str:
    """A time as a UTC ISO string with the Z suffix; a naive time means UTC, as everywhere in the repo."""
    utc = lenient_utc(value)
    if utc is None:
        raise ValueError(f"{what} is not a usable time: {value!r}")
    return iso_utc(utc)


def _optional_iso(value: datetime | None, what: str) -> str | None:
    return None if value is None else _iso(value, what)


def state_to_json(state: EngineState) -> dict:
    """The on-disk form of a state (the shape of ``emotion_state.json``); `state.saved_at` must be set."""
    if state.saved_at is None:
        raise ValueError("cannot write a state with no saved_at")
    emotions: dict[str, dict] = {}
    for name, emotion in state.emotions.items():
        emotions[name] = {
            "short": {"avg": emotion.short.avg, "weight": emotion.short.weight},
            "long": {"avg": emotion.long.avg, "weight": emotion.long.weight},
            "last_entry_day": emotion.last_entry_day,
            "snapshot": [
                [
                    day.day,
                    day.short,
                    day.long,
                    1 if day.felt else 0,
                    None if day.run is None else [day.run.start_day, day.run.direction, day.run.last_with_day],
                ]
                for day in emotion.snapshot
            ],
        }
    return {
        "saved_at": _iso(state.saved_at, "saved_at"),
        "reset_at": _optional_iso(state.reset_at, "reset_at"),
        "last_update": _optional_iso(state.last_update, "last_update"),
        "active_day": state.active_day,
        "persona_start_at": _optional_iso(state.persona_start_at, "persona_start_at"),
        "switch_over_at": _optional_iso(state.switch_over_at, "switch_over_at"),
        "queue_line": state.queue_line,
        "rows_through": state.rows_through,
        "repeats": {
            memory_id: {"first_at": _iso(entry.first_at, "first_at"), "count": entry.count}
            for memory_id, entry in state.repeats.items()
        },
        "emotions": emotions,
    }


def save_state(persona_dir: Path, state: EngineState, *, verify: bool = False) -> bool:
    """Write the state file atomically with three backups, compact. The caller holds the lock and has set `state.saved_at`.

    Does not change `state`. Serialization is strict: a non-finite number fails before any file is touched. With `verify=True`
    the file is read back under the same lock as the write and `.bak1` is put back if it is not what was written; the result is
    False then (the rebuild treats that as a failed check). Without it the result is True once the write is done.
    """
    return write_json(state_path(persona_dir), state_to_json(state), backups=_BACKUPS, indent=None, verify=verify)


class LoadedState(NamedTuple):
    """One load of the state file, immutable as a whole (the cache replaces it with one assignment).

    `state` is None when no copy was usable (`problem` says why). `source` is the copy it came from. `key` is the stat key
    ``(inode, mtime_ns, size)`` of the bytes parsed, the walk's ``fstat`` signature as returned (the triple that
    ``file_ops.stat_path(path).key`` gives for the same file), None when there is no state. `shifts` is what `EmotionEngine.read`
    works out and stores with `StateCache.store_shifts`. The `state` inside is a mutable object shared by everyone holding
    this tuple: treat it as read-only (a drain that folds reads its own copy with `read_state`).
    """

    key: tuple[int, int, int] | None
    state: EngineState | None
    source: str | None
    problem: str | None
    shifts: Mapping[str, int] | None = None

    @property
    def caught_up(self) -> bool:
        """False while the state came from a backup copy, which is as old as that copy's `saved_at` (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.8, recovery first)."""
        return self.source == "main"


def read_state(persona_dir: Path) -> LoadedState:
    """Read the state through the no-heal walk, every time, into fresh objects. Takes no lock and writes nothing."""
    path = state_path(persona_dir)
    walk = read_json_walk(path, expect=dict, validate=validate_state)
    if walk.problem == "unreadable":
        count_throttled(
            logger,
            f"engine_files:unreadable:{path}",
            logging.ERROR,
            "emotion state %s and its backups exist but none is usable; the state is unreadable",
            path,
        )
    if walk.problem is not None:
        return LoadedState(None, None, None, walk.problem)
    if walk.source != "main":
        count_throttled(
            logger,
            f"engine_files:backup:{path}:{walk.source}",
            logging.WARNING,
            "emotion state %s is not usable; loaded its older copy %s",
            path,
            walk.source,
        )
    state = state_from_json(walk.data, label=f"{path} ({walk.source})")
    return LoadedState(walk.signature, state, walk.source, None)


class StateCache:
    """The parsed state of one persona, kept in process and re-read when the main file's stat key changes (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.7).

    Only a state loaded from the main file is kept. A state from a backup, or none at all, is read again on the next call (a
    backup is a transient condition that the next drain's catch-up ends by writing a new main file). A hit needs the cached
    key to equal the main file's current stat key; the key is the signature of the bytes that were parsed, so a file replaced
    after the read does not match and is read again. The cache is replaced by one assignment of an immutable `LoadedState`,
    so a reader on another thread sees the old one or the new one, never half of one. The state inside is shared: read-only.
    """

    def __init__(self, persona_dir: Path) -> None:
        self._persona_dir = Path(persona_dir)
        self._loaded: LoadedState | None = None

    def get(self) -> LoadedState:
        loaded = self._loaded
        if loaded is not None:
            try:
                stat = file_ops.stat_path(state_path(self._persona_dir))
            except OSError:
                stat = None  # cannot tell: read again
            if stat is not None and stat.key == loaded.key:
                return loaded
        fresh = read_state(self._persona_dir)
        self._loaded = fresh if fresh.source == "main" else None
        return fresh

    def store_shifts(self, loaded: LoadedState, shifts: Mapping[str, int]) -> bool:
        """Keep `shifts`, worked out from `loaded`, with the cached state. False when the cache has moved on to another load.

        Two threads racing here can leave an older tuple in the cache; its key no longer matches the file, so the next
        `get` reads again.
        """
        if self._loaded is not loaded:
            return False
        self._loaded = loaded._replace(shifts=shifts)
        return True


# ================================================================================================
# Events: the queue's and the change log's one shape and one check
# ================================================================================================

# The kinds the queue knows. "start" is the start-only event Engine 1 added: it carries only `at`, the creation time of a
# committed row of a type that does not count, and keeps the persona's start date. It is a memory's own date event: like a
# commit and an assigned event it is APPLIED directly when the memory is written (`engine.EmotionEngine._apply_direct`), and
# only a showing and a change are appended to the queue. The check and the JSON form are shared by both routes.
_MEMORY_KINDS = frozenset({"commit", "assigned", "showing"})
_EVENT_KINDS = _MEMORY_KINDS | {"change", "start"}
# The range of a modifier: what `engine_utils.source_modifier` can make (1.0 for the user and her own, a peer or research memory
# from the floor up to floor + span). A stored modifier outside it is held to it; one far outside would otherwise fold
# to an infinite average and make every save of the state fail.
_MODIFIER_LOW = _MODIFIER_FLOOR
_MODIFIER_HIGH = _MODIFIER_FLOOR + _MODIFIER_SPAN


class InvalidEventError(ValueError):
    """An event the engine will not fold; the message says why (it is written to the log as it is)."""


def _event_number(raw: Mapping, key: str, default: float) -> float:
    if key not in raw:
        return default
    try:
        return _finite(raw[key], key)
    except ValueError as exc:
        raise InvalidEventError(str(exc)) from None


def event_from_json(raw: object) -> Event:
    """The `Event` of one queue or change-log record, or `InvalidEventError` saying why not.

    Invalid (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.13): not an object; an unknown `kind`; a missing `at`, or an `at` that `lenient_utc` rejects or that
    `day_of` cannot turn into a local day (such an event would fail the drain on every call, and a failure that comes from an
    event's own content must not repeat for ever, so the drop is the remedy); a commit, assigned or showing with no
    `memory_id` (a non-empty text); a commit or assigned with no emotion values; an emotion name that is not text or an amount
    that is not a finite number (a bool is not a number); a `modifier` or `climax`, when present, that is not a finite
    number; a `source`, when present, that is not text. A missing `climax` is 0.0. A missing `modifier` is 1.0; a finite one
    outside the range `source_modifier` can make (0.1 to 1.0) is not invalid: it is held to the range, with a warning line. The `modifier` of a change is not read (a change has weight 1, no repeat factor).
    """
    if not isinstance(raw, dict):
        raise InvalidEventError(f"not an object ({type(raw).__name__})")
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in _EVENT_KINDS:
        raise InvalidEventError(f"unknown kind {kind!r}")
    if "at" not in raw:
        raise InvalidEventError("missing at")
    at_text = raw["at"]
    at = lenient_utc(at_text) if isinstance(at_text, str) else None
    if at is None:
        raise InvalidEventError(f"at is not a time: {at_text!r}")
    try:
        day_of(at)
    except (OverflowError, OSError, ValueError) as exc:
        raise InvalidEventError(f"at cannot be turned into a local day: {at_text!r}") from exc
    if kind == "start":
        return Event(kind="start", at=at)
    memory_id: str | None = None
    if kind in _MEMORY_KINDS:
        memory_id = raw.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id:
            raise InvalidEventError(f"{kind} with no usable memory_id: {memory_id!r}")
    emotions_raw = raw.get("emotions", {})
    if not isinstance(emotions_raw, dict):
        raise InvalidEventError(f"emotions must be an object, found {type(emotions_raw).__name__}")
    emotions: dict[str, float] = {}
    for name, amount in emotions_raw.items():
        try:
            emotions[name] = _finite(amount, f"amount of {name}")
        except ValueError as exc:
            raise InvalidEventError(str(exc)) from None
    if kind in ("commit", "assigned") and not emotions:
        raise InvalidEventError(f"{kind} with no emotion values")
    if kind == "change":
        source = raw.get("source")
        if source is not None and not isinstance(source, str):
            raise InvalidEventError(f"source must be text, found {type(source).__name__}")
        return Event(kind="change", at=at, emotions=emotions, source=source)
    modifier = _event_number(raw, "modifier", 1.0)
    held = min(max(modifier, _MODIFIER_LOW), _MODIFIER_HIGH)
    if held != modifier:
        logger.warning(
            "emotion event %s %s: modifier %r is outside %s to %s, held to %s",
            kind,
            memory_id,
            modifier,
            _MODIFIER_LOW,
            _MODIFIER_HIGH,
            held,
        )
        modifier = held
    climax = _event_number(raw, "climax", 0.0) if kind == "commit" else 0.0
    return Event(kind=kind, at=at, memory_id=memory_id, emotions=emotions, modifier=modifier, climax=climax)


def event_to_json(event: Event) -> dict:
    """The record of an event (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.9): commit, assigned and showing carry a memory id, emotions and a modifier (a commit
    also its climax); a change carries its source and signed amounts and no modifier; a start-only event carries only `at`."""
    if event.kind not in _EVENT_KINDS:
        raise ValueError(f"unknown event kind {event.kind!r}")
    out: dict[str, Any] = {"kind": event.kind, "at": _iso(event.at, "at")}
    if event.kind == "start":
        return out
    if event.kind in _MEMORY_KINDS:
        out["memory_id"] = event.memory_id
        out["emotions"] = dict(event.emotions)
        out["modifier"] = event.modifier
        if event.kind == "commit":
            out["climax"] = event.climax
        return out
    if event.source is not None:
        out["source"] = event.source
    out["emotions"] = dict(event.emotions)
    return out


# ================================================================================================
# The queue
# ================================================================================================


class QueueRead(NamedTuple):
    """What `read_queue` found: the valid events after the position, and the position to keep for next time.

    `end_line` is the number of complete lines in the queue file (bad lines and blank lines count; a last line with no
    newline yet does not), which is the `from_line` of the next read once these events are folded.
    """

    events: list[Event]
    end_line: int


def read_queue(persona_dir: Path, *, from_line: int) -> QueueRead:
    """The valid events on the queue lines after line `from_line`, in file order, and the line count (`QueueRead`).

    `from_line` is the position the engine saved with its state (`EngineState.queue_line`): the lines up to it are already
    folded. A missing file is `([], 0)`. If `from_line` is larger than the file's line count, the queue was emptied since the
    position was saved, so it is read again from line 0 (`end_line` is the file's line count either way). A negative
    `from_line` is not a position: it is read as 0, with a log line (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.9).

    Accepted edge (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 6, "An unparseable queue line is dropped", applied per line; owner 2026-10-21 15:02): a line that is not valid JSON (a line of zero bytes left by an unclean machine stop, anything else)
    or that fails the event check is DROPPED with its own ordinary error log line naming the file, the line number and the
    reason, and the other lines are read. NO rebuild follows. What a drop loses is at most one turn's worth of events (the
    accepted loss of a queue entry, section 6); a rebuild would trade the long average's history older than 30 days for them. A line damaged in the
    middle takes anything glued to it along: nothing is salvaged. A dropped line still counts as a line, and so does a blank
    one. A last line with no newline yet is a torn write, or an append in flight: it is not returned, not counted and not
    logged, and is read whole once it is complete (if a crash left it torn, the next append starts a new line, and it is then
    a complete, bad line). The read itself writes nothing: the bad lines stay in the file (and are logged again by a read from
    before them) until the drain empties it.

    A fault in the READING (permissions, a sharing violation, a directory in the file's place) is not a drop: it raises, the
    queue file is left alone, and the caller's failure handling keeps the queue for the next drain (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.13).

    The file is read by ``json_io.read_jsonl_from`` (one bytes read, the one line classifier); this function adds the event
    check and the log lines, which name the physical line number and the reason and echo no content.
    """
    path = queue_path(persona_dir)
    if from_line < 0:
        count_throttled(
            logger,
            f"engine_files:queue_from_line:{path}",
            logging.ERROR,
            "emotion queue %s: the position %d is below 0, read as 0 (the queue is read from its start)",
            path,
            from_line,
        )
        from_line = 0
    result = read_jsonl_from(path, from_line)
    if from_line > result.end_line:
        result = read_jsonl_from(path, 0)
    events: list[Event] = []
    for record in result.records:
        if record.reason is not None or record.data is None:
            logger.error("emotion queue %s: line %d dropped (%s)", path, record.line, record.reason)
            continue
        try:
            events.append(event_from_json(record.data))
        except InvalidEventError as exc:
            logger.error("emotion queue %s: line %d dropped (%s)", path, record.line, exc)
    return QueueRead(events, result.end_line)


def append_event(persona_dir: Path, event: Event) -> bool:
    """Append an event as one line at the end of the queue (the caller holds the lock). True if it was written.

    Earlier lines are never read or rewritten, so an append costs the same at any depth of the queue. Best effort, by design
    (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 2.13): a failure (a full disk, a folder that is a file, a value that cannot be encoded) is logged and gives False, and
    never raises into the caller. An OSError is logged by ``append_jsonl`` (a throttled warning); a value that cannot be
    encoded gets an error line here. No fsync, no backups.

    Accepted edge (`docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md` section 6, "A lost queue entry"): an entry lost to a failed append, or damaged by an unclean machine stop, is an accepted minor edge. It is not
    recovered; a damaged line is dropped on read (once it is a complete line), and takes any record glued to it along.
    """
    return _append_line(lambda: queue_path(persona_dir), event, "queue")


def empty_queue(persona_dir: Path) -> bool:
    """Empty the queue file, atomically: True once it is empty, False (with a throttled log line) when it could not be. Never raises.

    ``write_jsonl(path, [], fsync=False)`` replaces the file with an empty one; the engine never deletes it. A failure leaves
    the queue exactly as it was. The caller holds the lock and has folded every line it read; the position it holds
    (``EngineState.queue_line``) is the caller's to reset.
    """
    path = queue_path(persona_dir)
    try:
        write_jsonl(path, [], fsync=False)
    except OSError as exc:
        warn_throttled(
            logger,
            f"engine_files:empty_queue:{path}:{exc}",
            "emotion queue %s could not be emptied (%s); the queue is kept as it is",
            path,
            exc,
        )
        return False
    return True


# ================================================================================================
# The change log
# ================================================================================================


def _name_to_day(name: str) -> str | None:
    """The day of a change-log file name ``YYYY-MM-DD.jsonl``, or None for any other name."""
    stem, dot, suffix = name.rpartition(".")
    if not dot or suffix != "jsonl":
        return None
    try:
        return _day(stem, "file name")
    except ValueError:
        return None


def _append_line(path_of: Callable[[], Path], event: Event, what: str) -> bool:
    """Append `event` as one line to the file `path_of()` names, never raising: the shared failure rule of the queue and the log.

    The caller holds the lock (``lock="held"``). A failure of the append itself is logged by ``append_jsonl``; a value that
    cannot be encoded, or a time that cannot be dated (the path is built inside the guard), gets an ordinary error line here.
    """
    try:
        # fsync off: the engine's design (section 2.9) chose no fsync per append; asked of ToT on #618's PR.
        written = append_jsonl(path_of(), event_to_json(event), lock="held", fsync=False, on_error="log")
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        logger.error("emotion %s: an event at %s was not recorded (%s: %s)", what, event.at, type(exc).__name__, exc)
        return False
    return written is not None


def append_change(persona_dir: Path, event: Event) -> bool:
    """Append a change event as one line to the file of its local day (the caller holds the lock). True if it was written.

    Best effort, by design: a failure (a full disk, a value that cannot be encoded, a time that cannot be dated) gets an
    ordinary error log line and False, and never raises into the caller; the change was already queued (the caller appends here
    only after the queue append succeeded), so it still folds through the queue.
    Only a `change` event belongs here (anything else is a ValueError).
    """
    if event.kind != "change":
        raise ValueError(f"the change log holds change events, not {event.kind!r}")
    return _append_line(lambda: changes_dir(persona_dir) / f"{day_of(event.at)}.jsonl", event, "change log")


def list_change_files(persona_dir: Path) -> list[str]:
    """The names (only) of the change-log files, oldest first: regular files named ``YYYY-MM-DD.jsonl`` for a real day.

    The folder is listed with ``file_ops.list_dir``; a name that matches the pattern is kept only when
    ``file_ops.stat_path`` says it is a regular file, so a folder named like a day is never listed (and never deleted). An
    entry that vanished (no stat) or whose stat fails is skipped. A missing folder is empty. A folder that cannot be listed is
    logged and treated as missing.
    """
    folder = changes_dir(persona_dir)
    try:
        entries = file_ops.list_dir(folder)
    except OSError as exc:
        logger.error("emotion change log folder %s cannot be listed (%s); treated as missing", folder, exc)
        return []
    names: list[str] = []
    for name in entries:
        if _name_to_day(name) is None:
            continue
        try:
            stat = file_ops.stat_path(folder / name)
        except OSError:
            continue
        if stat is not None and stat.is_file:
            names.append(name)
    return sorted(names)


def prune_changes(persona_dir: Path, anchor_day: str | None) -> list[str]:
    """Delete whole change-log files whose day is before the label 30 calendar days before `anchor_day`; return their names.

    `anchor_day` is the active day as it stood BEFORE the drain's own day closes (the caller reads it at the start of the
    drain), so a system clock set far forward cannot delete the folder in the drain that jumps the active day. Nothing is
    deleted when it is None (unset). The length is ``dev_constants.EMOTION_LONG_HALF_LIFE_D``. A file on or after the
    boundary stays. Only regular files named ``YYYY-MM-DD.jsonl`` are ever touched. A delete that fails is logged and the
    rest go on; the file is tried again at the next day close. The caller holds the lock.
    """
    if anchor_day is None:
        return []
    try:
        boundary = (date.fromisoformat(_day(anchor_day, "anchor day")) - timedelta(days=dev_constants.EMOTION_LONG_HALF_LIFE_D)).isoformat()
    except (ValueError, OverflowError) as exc:
        logger.error("emotion change log: no files pruned, the active day %r is unusable (%s)", anchor_day, exc)
        return []
    deleted: list[str] = []
    for name in list_change_files(persona_dir):
        day = _name_to_day(name)
        if day is None or day >= boundary:
            continue
        try:
            file_ops.delete_file(changes_dir(persona_dir) / name)
        except OSError as exc:
            logger.error("emotion change log: %s could not be deleted (%s); it is tried again at the next day close", name, exc)
            continue
        deleted.append(name)
    return deleted


def read_changes(persona_dir: Path, since: datetime) -> list[Event]:
    """The change events dated at or after `since`, for the replay of a rebuild or a catch-up.

    Reads the files in date order starting at the day BEFORE ``day_of(since)`` (files are named by the local day at write
    time, so a time-zone change could put a first in-window line in an earlier-named file) and keeps the lines dated at or
    after `since`; the caller applies the end of its window and folds in time order. The log is a crash-recovery record and
    its absence or damage is normal: a missing file anywhere in the series, a missing folder, a file not named
    ``YYYY-MM-DD.jsonl``, an unlistable folder (logged), a torn or unparseable line (including one the JSON parser cannot
    load, such as an integer of more than 4300 digits, skipped with a warning naming the file and line) and a file that
    cannot be read in full (what was read is used) add nothing and are not errors. A line that fails the event check (including one whose `kind` is not a known text), or is not a change, is dropped
    with an error log line. The reader repairs nothing.
    """
    when = lenient_utc(since)  # a naive time means UTC, as everywhere in the repo
    if when is None:
        raise ValueError(f"since is not a usable time: {since!r}")
    since = when
    start = (date.fromisoformat(day_of(since)) - timedelta(days=1)).isoformat()
    folder = changes_dir(persona_dir)
    events: list[Event] = []
    for name in list_change_files(persona_dir):
        day = _name_to_day(name)
        if day is None or day < start:
            continue
        path = folder / name
        for _text, record in iter_jsonl_raw(path):  # a line the parser cannot load is skipped; an I/O failure ends the file there
            try:
                kind = record.get("kind") if isinstance(record, dict) else None
                if isinstance(kind, str) and (kind in _MEMORY_KINDS or kind == "start"):  # before the check, so a held-modifier line is not logged for it
                    raise InvalidEventError(f"a {kind} event is not a change")
                event = event_from_json(record)
                if event.kind != "change":
                    raise InvalidEventError(f"a {event.kind} event is not a change")
            except InvalidEventError as exc:
                logger.error("emotion change log %s: a line was dropped (%s)", path, exc)
                continue
            if event.at >= since:
                events.append(event)
    return events


# ================================================================================================
# Whether the persona was switched over
# ================================================================================================


def switch_over_state(loaded: LoadedState) -> bool | None:
    """Whether a persona is switched over, read off one load of its state: True, False, or None when that cannot be told.

    There is no switch file: the first rebuild at start-up writes `switch_over_at` into the state file, in its one verified
    write, and every later save keeps it, so "switched" is the loaded state holding it (any copy the walk accepted, main or a
    backup, so a persona recovered from a backup copy stays switched). True: the loaded state holds `switch_over_at`. False: it
    holds none, or there is no state file at all. None: the state exists but no copy can be read (`loaded.problem` is
    "unreadable"): that cannot say either way. Owner 2026-11-05: "If the system can't tell if its switched, assume its not."
    The engine reads None as not switched and never remembers it. The start-up does not skip on it (Planning B31 SW4, owner
    S79): it decides from the database's x10 record (`nudge_scale_fix`) and runs the first-mode rebuild, whose state write
    replaces the unreadable state like any other (owner S88; its normal backup rotation, nothing moved aside); the drain and the
    direct apply skip their pass and keep the queue until then.

    Accepted edges (Planning B25 SW2, P4). (a) Every copy of the state file lost or unreadable, live and backups: the persona
    reads as not switched until the first rebuild runs again at the next start, and that run records a later switch-over time. The x10 fix's progress note in `memories.db` (`PRAGMA user_version`) stops it
    multiplying anything again on the re-run. (b) The live state is corrupted right after the first rebuild while `.bak1` is still the pre-switch
    state: the backup has no `switch_over_at`, so the persona reads as not switched until the first rebuild runs again (the
    same as (a)). Narrowed (Planning B28 Q18): a process that has already SEEN `switch_over_at` and loads a backup without it
    puts the value it saw back before its drain saves (`EmotionEngine._run_apply`), so (b) still applies to a new process
    that starts after the live state is corrupted. (c) A brain built from main before Engine 6 merges (a rolled-back brain-main
    overlay, a source install on main, or a release cut from main before the merge; no published release has it) runs a shadow
    engine that rewrites the state without `switch_over_at` or `rows_through` and keeps `saved_at` fresh. A persona rolled back
    to it and forward again reads as not switched, so the first rebuild runs again from the 30-day window and records a new
    switch-over time, which restarts the fade clock: the rebuilt mood reads stronger than her real state for a while. Nothing
    durable records the original switch time (TRIO HR1-C, accepted).
    """
    if loaded.state is not None:
        return loaded.state.switch_over_at is not None
    return None if loaded.problem == "unreadable" else False
