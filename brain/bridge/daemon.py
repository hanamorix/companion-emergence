"""SP-7 bridge daemon orchestration — process spawn, stop, status.

Public surface used by CLI handlers in brain.cli:
  cmd_start(args)    -> int
  cmd_run(args)      -> int
  cmd_stop(args)     -> int  (kill_pre_pr=False: the start's stop before the rebuild never hard-kills a pre-PR bridge)
  cmd_restart(args)  -> int
  cmd_status(args)   -> int
  cmd_tail(args)     -> int
  cmd_tail_log(args) -> int

Used by the commands that touch the engine:
  run_rebuild_step(persona_dir) -> int | None  (None: go on; else the exit code)

Internal:
  spawn_detached(persona_dir, idle_shutdown_seconds, client_origin, log_path) -> pid
"""

from __future__ import annotations

import codecs
import errno
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx

from brain.bridge import respawn_guard, start_backoff, state_file
from brain.bridge.shutdown import REASON_USER_STOP
from brain.health.log_rotation import rotate_rolling_size
from brain.paths import RUNTIME_LOG_BACKUPS, RUNTIME_LOG_MAX_BYTES

logger = logging.getLogger(__name__)

LOCKFILE = "bridge.json.lock"

# S50/S56: real OS-level lock (like brain.utils.file_lock's sidecar pattern),
# held for the bridge's whole process life and released by the OS itself on
# crash or reboot — no pid-alive / age / health-probe staleness guessing.
_IS_WINDOWS = sys.platform.startswith("win")

if _IS_WINDOWS:
    # msvcrt is a Windows-only stdlib module; importing on POSIX would fail
    # at module load (mirrors brain.utils.file_lock's guard).
    import msvcrt
else:
    import fcntl

# Windows only: the byte range msvcrt.locking() locks/unlocks. Windows
# mandatory byte-range locking blocks ANY overlapping I/O from another
# handle — including a plain read — not just writes (unlike POSIX flock,
# which is advisory and never blocks a read). A far, fixed offset well
# beyond the file's real (tiny) content means a normal whole-file read of
# that real content (which ends long before this offset) never overlaps
# the locked range, so the pid at offset 0 stays plainly readable while
# the lock is held (C18c) — no special "read only this sub-region" contract
# for readers, and the pid sits at the SAME offset on both platforms.
# msvcrt.locking locks a real byte range via LockFileEx even when nothing
# has ever been written there; it does not require file content to exist
# at that offset, and an os.ftruncate() to a smaller size afterward does
# not release or move the lock (the lock is independent of current EOF).
_WINDOWS_LOCK_OFFSET = 1 << 20  # 1 MiB

# msvcrt.locking(LK_LOCK) raises OSError with this errno when the byte is still locked after its own 10 tries.
_EDEADLOCK = getattr(errno, "EDEADLOCK", errno.EDEADLK)


def acquire_lock(persona_dir: Path, *, blocking: bool = False) -> int | None:
    """Take the bridge's OS-level lock for the whole process life.

    Same pattern as brain.utils.file_lock: open with O_CREAT (never O_EXCL —
    the file itself is never recreated, only its lock contended for) and take
    a non-blocking flock/msvcrt lock on that fd. A crash or reboot releases
    the OS lock instantly, so there is no pid-alive / age / health-probe
    staleness logic left to guess with (S50) — the lock IS the liveness
    signal.

    Returns the open fd (the caller must keep it open for the lock's
    duration and pass it to release_lock) on success, or None if another
    live process holds the lock. With ``blocking=True`` it waits until the lock
    is free instead (POSIX: a blocking flock; Windows: msvcrt's own 10 x 1 s
    LK_LOCK retry, repeated while the lock is still held; any other lock error
    returns None), as `nell supervisor run` does for OS services.

    The pid is written into the file for information only (S50) — never
    read back to make a decision here. It sits at offset 0 on BOTH
    platforms: on Windows the locked byte range is at a far fixed offset
    (`_WINDOWS_LOCK_OFFSET`, see its own comment) well past the file's real
    content, so a plain whole-file read never overlaps it; on POSIX flock
    is advisory over the whole file regardless, so a reader was never
    blocked there either way.
    """
    path = persona_dir / LOCKFILE
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
    try:
        if _IS_WINDOWS:
            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            if blocking:
                while True:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # retries 10 x 1 s itself, then raises
                        break
                    except OSError as exc:
                        # Only "still held after its 10 tries" (EDEADLOCK) is contention; any other error (a bad fd,
                        # an access error) would repeat forever, so it ends the wait (review O-3).
                        if exc.errno != _EDEADLOCK:
                            raise
                        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            else:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return None

    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, str(os.getpid()).encode())
    return fd


def lock_holder(persona_dir: Path) -> int | None:
    """The pid written in the bridge lock while another process holds it, else None (free, or a holder whose pid
    cannot be read). A probe for the kill helper (C45 i): a non-blocking lock attempt on its own fd, released at
    once; it never writes the file."""
    path = persona_dir / LOCKFILE
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            if _IS_WINDOWS:
                os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            return None  # nobody holds it
        except OSError:
            pass  # held by another process
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 32)
    except OSError:
        return None
    finally:
        os.close(fd)
    try:
        return int(raw.decode("ascii", "replace").strip())
    except ValueError:
        return None


def release_lock(persona_dir: Path, fd: int) -> None:
    """Unlock and close the fd. Never unlinks the file (S56) — unlinking

    would let a second process create a NEW file under the same name and
    lock THAT one, so two processes could each believe they hold "the"
    bridge lock on two different inodes."""
    try:
        if _IS_WINDOWS:
            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _kill_if_alive(pid: int, sig: int) -> bool:
    """``os.kill(pid, sig)``, tolerating "already dead" identically on both
    platforms. Returns True if the pid was alive and got signaled, False if
    it was already dead.

    POSIX: a dead pid raises ``ProcessLookupError``. Windows: ``os.kill``'s
    ``TerminateProcess``-based implementation instead raises a plain
    ``OSError`` with ``winerror == 87`` (ERROR_INVALID_PARAMETER) for an
    already-exited pid — a documented CPython-on-Windows quirk, NOT
    ``ProcessLookupError`` — so it must be checked for explicitly rather
    than assumed covered by the POSIX exception type. (CI 2026-09-26:
    windows-latest hit exactly this, unhandled, inside cmd_start's
    readiness-timeout orphan-kill path.)
    """
    try:
        # windows-lint: ok -- the one hard-kill helper: an orphan after a failed start, or the
        # fallback when the graceful shutdown endpoint failed. TerminateProcess is intended there.
        os.kill(pid, sig)  # windows-lint: ok
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        if _IS_WINDOWS and getattr(exc, "winerror", None) == 87:
            return False
        raise


def _path_eq(p1: str, p2: str) -> bool:
    return p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2)


def bridge_python() -> tuple[str, dict[str, str] | None]:
    """The interpreter (and env, or None to inherit) to spawn a bridge child with,
    such that the spawned process's pid IS the runner's own ``os.getpid()``.

    Windows venv: ``sys.executable`` (``.venv\\Scripts\\python.exe``) is the
    venv *redirector* (``venvlauncher.exe``, used by both ``python -m venv``
    and uv), which starts the real base interpreter as a SECOND process. So
    ``Popen(...).pid`` is the redirector's pid, while the runner writes its
    own (different) ``os.getpid()`` into bridge.json. ``cmd_start``'s
    readiness check (``s.pid == pid``) could then never match: a healthy
    bridge was reported as failed after the full 50s wait and its
    redirector killed. Same fix as the stdlib's own
    ``multiprocessing.popen_spawn_win32``: launch ``sys._base_executable``
    directly and pass the venv via ``__PYVENV_LAUNCHER__``, exactly what the
    redirector itself would have done, minus the extra process.

    Everywhere else (POSIX, where a venv python is a symlink/copy with no
    redirector; or a non-venv Windows runtime such as the bundled
    python-build-standalone one) this is plain ``sys.executable``, env
    inherited.
    """
    base = getattr(sys, "_base_executable", None)
    if _IS_WINDOWS and base and not _path_eq(sys.executable, base):
        env = os.environ.copy()
        env["__PYVENV_LAUNCHER__"] = sys.executable
        return base, env
    return sys.executable, None


def spawn_detached(
    persona_dir: Path,
    idle_shutdown_seconds: float | None,
    client_origin: str,
    log_path: Path,
) -> int:
    """Spawn the bridge server in a detached process. Returns child pid —
    the runner's own pid on every OS (see ``bridge_python``).

    The Popen handle is kept in ``_spawned_children`` (keyed by that pid) so
    ``cmd_start`` can notice promptly, via ``Popen.poll()``, that the child
    already exited (e.g. lost the bridge lock) instead of waiting out its
    whole readiness window. ``poll()`` and not a pid-alive probe: on POSIX an
    exited-but-unreaped child is a zombie that ``os.kill(pid, 0)`` still
    reports alive, and ``poll()`` is also what reaps it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab")  # noqa: SIM115
    python, env = bridge_python()
    popen_extra: dict[str, object] = {} if env is None else {"env": env}
    cmd = [python, *_runner_argv(persona_dir, idle_shutdown_seconds, client_origin)]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            **popen_extra,
        )
        _spawned_children[proc.pid] = proc
        return proc.pid
    finally:
        log_fh.close()


def _rotate_bridge_log(log_path: Path) -> None:
    """Outside launchd nothing else bounds bridge-<persona>.log (#559). cmd_start calls this while it still holds the
    bridge lock, so no running bridge (a runner holds that lock for its whole life) is writing the file. Not airtight:
    another starter's child, spawned after that starter released the lock but before the child took it, can be writing.
    Its stdout appends (O_APPEND), so the copy-then-truncate costs at most the lines it wrote between the two, and only
    when the log is over the cap. A failed rotation never blocks a start."""
    try:
        rotate_rolling_size(log_path, RUNTIME_LOG_MAX_BYTES, RUNTIME_LOG_BACKUPS)
    except Exception:  # noqa: BLE001
        logger.warning("could not rotate %s; starting the bridge anyway", log_path, exc_info=True)


# pid -> Popen for children spawn_detached started and cmd_start has not yet
# finished waiting on (cmd_start pops its own entry). See spawn_detached.
_spawned_children: dict[int, subprocess.Popen] = {}


def _runner_argv(
    persona_dir: Path, idle_shutdown_seconds: float | None, client_origin: str
) -> list[str]:
    """The bridge child's argv after the interpreter (see spawn_detached)."""
    argv = [
        "-P",  # -m would put the caller's cwd (maybe a checkout's brain/) on sys.path
        "-m",
        "brain.bridge.runner",
        "--persona-dir",
        str(persona_dir),
        "--client-origin",
        client_origin,
    ]
    if idle_shutdown_seconds is not None:
        argv += ["--idle-shutdown-seconds", str(idle_shutdown_seconds)]
    return argv


# The owner's wording list (S101, S102): the same lines as the app's start-up screen (useRestartBridge.ts engineProgressLine).
_REBUILD_STARTED_LINE = "Updating your memories. This can take a few minutes."
_REBUILD_RESCALING_LINE = "Updating the scale of your memories."
_REBUILD_PREPARING_LINE = "Getting ready to update your memories."


def _say(line: str) -> None:
    """One line to stderr. A stderr that is gone or a print that fails (a broken pipe) is ignored: a display line never
    changes what the step does."""
    try:
        if sys.stderr is not None:
            print(line, file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass


def _stderr_is_terminal() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _usable_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _rebuild_progress_line(record: dict | None) -> str | None:
    """The window's line for a progress record: the scale update has no count, a record with no usable total is "getting ready",
    a count is "N of M (P%)" (N held between 0 and M). None when there is no record. Same rule as the app's
    ``engineProgressLine`` for the whole numbers the engine writes (a fractional count is cut to a whole number here)."""
    if record is None:
        return None
    if record.get("phase") == "rescaling":
        return _REBUILD_RESCALING_LINE
    done, total = record.get("done"), record.get("total")
    if not (_usable_number(done) and _usable_number(total)):
        return _REBUILD_PREPARING_LINE
    done, total = int(done), int(total)
    if total <= 0:
        return _REBUILD_PREPARING_LINE
    counted = min(max(done, 0), total)
    return f"Updating your memories: {counted} of {total} ({counted * 100 // total}%)."


def _rebuild_under_lock(persona_dir: Path) -> int | None:
    """The start's rebuild step (S107, HS5 of #633), run with the start lock held. Returns None to go on, or the exit code to
    return: 1 when another running command could not be ended (nothing else ran).

    (1) Every other running command (``engine_files.live_command_runs``) is ended with ``kill.kill_command``, which asks the
    process for its identity first; a command that is not ended (survivors, an identity that could not be confirmed, no token, an
    exception) ends the step with one plain line naming its pid. This is the one place a start acts on a running command.
    (2) The rebuild is its own process, ``nell supervisor rebuild``, waited for with no limit (S101: the startup waits until the
    rebuild is finished). It prints the lines below to stderr only when stderr is a terminal; the loop sleeps through
    ``stop_watch._clock`` so the app's captured stderr of a failed start is not flooded. A return code other than 0, or an OSError
    from the spawn, is one failed rebuild: one line, one log entry, and the step ends normally on the state saved before (S107). Any
    BaseException (Ctrl-C included) kills and waits for the child and propagates, as ``subprocess.run`` does.

    The runtime log is attached for the step (transient, only when no persistent handler for the log is attached), so the lines
    logged here reach it; ``runner`` is imported here, not at module level, so a start with no rebuild never loads the bridge stack.

    Accepted edges (P4, R2): a persona whose rebuild keeps failing or skipping (S61's unreadable state with a failing integrity
    check) is needed at every start, so each start with no bridge running repeats the child (twice for a service-managed persona: the
    start's step, then the service run's) and goes on with the saved state, and each such run first ends every other running command,
    a ``nell chat --no-bridge`` session included; each run logs why. ``nell supervisor stop`` typed during the step finds no bridge,
    returns 0, and the start then launches the bridge. A child left by a dropped start holds the engine lock, so the new child waits
    for that lock in the existing lock wait and then finds the work done; a lock held by something that is not a rebuild makes the
    start wait with no limit and no stop check (S101; the child shows its "rebuilding" record meanwhile). No console flag is added on
    Windows (the real-child test runs on every leg). A command that goes on after a failed rebuild runs on an engine state that may
    lag until the next start retries (REVIEW, as the bridge does).
    """
    from brain.bridge import kill, runner
    from brain.emotion import engine_files

    transient = None if runner._runtime_log_attached(persona_dir) else runner._attach_transient_runtime_log(persona_dir)  # noqa: SLF001
    try:
        for record in engine_files.live_command_runs(persona_dir):
            pid, token = record.get("pid"), record.get("pid_token")
            try:
                result = kill.kill_command(pid, token if isinstance(token, str) else None)
                ended = result.succeeded
            except Exception as exc:  # noqa: BLE001 - a command that cannot be ended ends the step like a failed stop
                logger.warning("could not end the running command %s: %s: %s", pid, type(exc).__name__, exc)
                ended = False
            else:
                if ended and result.outcome == "killed":
                    logger.info("ended the running command %s before the rebuild", pid)
                elif not ended:
                    logger.warning("could not end the running command %s: %s", pid, result.message)
            if not ended:
                _say(f"Another command (process {pid}) is still running and could not be stopped. Nothing was updated. Try again when it has finished.")
                return 1
        _run_rebuild_child(persona_dir)
        return None
    finally:
        runner._detach_runtime_log(transient)  # noqa: SLF001


def _run_rebuild_child(persona_dir: Path) -> None:
    """Spawn ``nell supervisor rebuild`` and wait for it (HS5 (2)); see ``_rebuild_under_lock``. Never raises for a failed child."""
    from brain.bridge import stop_watch
    from brain.emotion import engine_files

    python, env = bridge_python()
    # "-P" as _runner_argv has it; the persona by name, from the same home as ours (the environment is inherited).
    argv = [python, "-P", "-m", "brain.cli", "supervisor", "rebuild", "--persona", persona_dir.name]
    show = _stderr_is_terminal()
    try:
        child = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
        )
    except OSError as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("the rebuild child for %s could not be started (%s); going on with the saved state", persona_dir.name, reason)
        _say(f"Updating your memories could not be started ({reason}). Going on with your memories as they were saved.")
        return
    try:
        if show:
            _say(_REBUILD_STARTED_LINE)
        last = _REBUILD_STARTED_LINE
        while child.poll() is None:
            if show:
                try:
                    line = _rebuild_progress_line(engine_files.read_engine_progress(persona_dir))
                    if line is not None and line != last:
                        last = line
                        _say(line)
                except Exception:  # noqa: BLE001 - a display glitch never touches the child
                    pass
            stop_watch._clock.sleep(stop_watch.POLL_S)  # noqa: SLF001
    except BaseException:
        try:
            child.kill()
        except OSError:  # the child is already going away; still wait for it, and let the original exception through
            pass
        child.wait()
        raise
    code = child.returncode
    if code != 0:
        logger.warning("the rebuild child for %s exited with code %s; going on with the saved state", persona_dir.name, code)
        _say(f"Updating your memories did not finish (exit code {code}). Going on with your memories as they were saved.")


def _rebuild_needed(persona_dir: Path, bridge_running: bool) -> bool:
    """Does a start (or a command) have to run the rebuild step first (S107, HS4 of #633)?

    With no bridge recorded running it is ``engine.start_rebuild_needed``: the persona is not switched over, or a roll-forward is due.
    With a bridge running it is only "the persona is not live": while a bridge runs, the roll-forward detector cannot be read as a need,
    because ``engine_rebuild.rolled_forward`` is true for any row above ``rows_through`` created after ``saved_at`` and nothing says who
    wrote the row, and a running bridge's own writes meet it (the row is committed before the engine applies it, the window S99
    accepted). A true roll-forward (rows an older brain wrote) is found at a start with no bridge running. REVIEW, the direction: this
    departs from S107's words in one case, a roll-forward need under a running bridge; the alternative (stop the bridge on it too)
    would stop and restart a healthy bridge, losing its in-flight work, whenever a start is typed over it inside that window.

    A check that raises is logged (type and reason, to the runtime log) and answers False: the start goes on. The bridge stack is
    imported only in that branch, so the no-rebuild path never loads it; its transient log handler is attached only when no persistent
    one is, and always detached again.
    """
    try:
        from brain.emotion import engine

        if not bridge_running:
            return engine.start_rebuild_needed(persona_dir)
        return not engine.engine_for(persona_dir).is_live()
    except Exception as exc:  # noqa: BLE001 - a failed check never stops a start
        from brain.bridge import runner

        transient = None if runner._runtime_log_attached(persona_dir) else runner._attach_transient_runtime_log(persona_dir)  # noqa: SLF001
        try:
            logger.warning(
                "could not check whether the emotion engine needs a rebuild for %s (%s: %s); going on with the saved state",
                persona_dir.name,
                type(exc).__name__,
                exc,
            )
        finally:
            runner._detach_runtime_log(transient)  # noqa: SLF001
        return False


def _stop_for_rebuild(persona_dir: Path, args) -> int:  # noqa: ANN001
    """Stop the running bridge so the rebuild can run (S107, HS4): the restart's stop (``cmd_restart``; reason None, so a queued
    service run is not held back), with ``kill_pre_pr=False``: a bridge with no identity token is asked and waited for, but never
    hard-killed (S110: no automatic hard kill; the app keeps its own hard-exit offer). A non-zero result ends the start before any
    rebuild: one plain line, and the start returns the code. Nothing is hard-killed here: a failed request falls back to SIGTERM on
    POSIX, the wait is the stop watch (which offers the hard exit after 180 s in a terminal and kills only on the user's "y"), and
    on Windows an unreachable endpoint is refused unless the user runs ``nell supervisor stop --kill``.

    Accepted edges (P4, R2): the app drops a start after 60 s with no rebuild record (lib.rs ``wait_for_start``, ``kill_on_drop``),
    which ends the start command, not the stop; no data is lost, and at launch on an unconfirmed overlay a failed start reaches
    ``recoverUnhealthyBrain`` and undoes the update (brainRecovery.ts), as HR5 (6) and (7) recorded. A stop that answers 0 while a
    bridge is still there (an UNKNOWN identity) is not caught here: a live bridge holds the start lock, so ``cmd_start``'s lock
    refuses with exit 2. A queued service run that wakes between the stop and ``cmd_start``'s lock takes it: this start then
    returns 2 and that run does the rebuild (``cmd_run``).
    """
    rc = cmd_stop(args, reason=None, kill_pre_pr=False)
    if rc != 0:
        _say(
            "The running bridge did not stop, so nothing was updated and no new bridge was started. "
            "A bridge from a previous version may still be stopping. "
            f'If it is stuck, end it with "nell supervisor stop --persona {persona_dir.name} --kill", then start again.'
        )
    return rc


def run_rebuild_step(persona_dir: Path) -> int | None:
    """The rebuild step for the commands that touch the engine (S107, HS6 (c) of #633): None to go on, or the exit code to return.

    With a bridge recorded running it does nothing (the command runs beside the bridge, as today). Otherwise, if a rebuild is needed
    it takes the start lock WITHOUT waiting: a start or another rebuild holds it, so the command is refused at once with exit 2 and
    one plain line (REVIEW, the reading: S102's "everything else waits" and S36's 5 s wait become "refused at once, retry"; a
    blocking acquire would also wait for a bridge's whole life, because the runner holds this lock from before its state file
    shows it). It runs ``_rebuild_under_lock`` with the lock held and releases it; it starts no bridge.
    """
    if state_file.is_running(persona_dir):
        return None
    if not _rebuild_needed(persona_dir, False):
        return None
    fd = acquire_lock(persona_dir)
    if fd is None:
        _say("A start or an update of your memories is under way. Try again when it has finished.")
        return 2
    try:
        return _rebuild_under_lock(persona_dir)
    finally:
        release_lock(persona_dir, fd)


# S57: cmd_start's refusal paths keep their existing wording, byte-identical.
_LOCK_HELD_MSG = "bridge already starting (lockfile held)"


def _refuse_already_running(persona_dir: Path, out: dict | None) -> int:
    """cmd_start's state-file "already running" refusal (S57 wording). Returns 2."""
    cur = state_file.read(persona_dir)
    print(f"bridge already running on port {cur.port} (pid {cur.pid})", file=sys.stderr)
    if out is not None and cur is not None and cur.pid is not None and cur.port is not None:
        out["readiness"] = BridgeReadiness(pid=cur.pid, port=cur.port, auth_token=cur.auth_token)
    return 2


@dataclass
class BridgeReadiness:
    """Verified-live snapshot of the freshly-spawned (or already-running) bridge.

    Returned via the `out` dict mutation pattern so cmd_start's int return
    code stays compatible with argparse handlers, while callers that need
    to immediately connect (the chat REPL) can grab the verified
    pid/port/auth_token directly without re-reading state_file. The
    re-read was the race in the 2026-05-05 audit-3 Bug B: state_file
    can be rewritten by the bridge supervisor between cmd_start's
    /health verification and the caller's read.
    """

    pid: int
    port: int
    auth_token: str | None


def cmd_start(args, *, out: dict | None = None) -> int:
    """Spawn the bridge daemon. Returns 0 on success, 2 on already-running, 1 on error.

    On success (or already-running), if `out` is provided, populates
    `out["readiness"]` with a BridgeReadiness carrying the verified
    pid/port/auth_token. Callers that need an immediate WS/HTTP connection
    should use that instead of re-reading state_file.

    The one-shot rebuild (S107, HS6 (a) of #633) runs here, before the bridge: a bridge recorded running is first stopped when a
    rebuild is needed (``_stop_for_rebuild``; a stop that does not finish returns its code and starts nothing), then, under the start
    lock, ``_rebuild_under_lock``; a step that returns a code releases the lock and returns it. A start over a running bridge with no
    rebuild needed, and an ``attach`` start (``nell chat``'s already-running branch: ``args.attach``) over a running bridge, are
    refused with exit 2 as before, an attach start before any need check.
    """
    from brain.paths import get_log_dir, get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    running = state_file.is_running(persona_dir)
    if running and getattr(args, "attach", False):
        return _refuse_already_running(persona_dir, out)
    needed = _rebuild_needed(persona_dir, running)
    if needed and running:
        stop_rc = _stop_for_rebuild(persona_dir, args)
        if stop_rc != 0:
            return stop_rc
    elif running:
        return _refuse_already_running(persona_dir, out)

    # Hold the lock through the start-up steps below (so a concurrent starter
    # gets "lock held" instead of starting a second bridge), then RELEASE it
    # right before spawning. The bridge PROCESS (the detached runner, see
    # runner.main) re-acquires it and holds it for its lifetime, so a stale
    # record is told apart from a live bridge. cmd_start must NOT hold the
    # lock across the spawn, or the child can't acquire it and dies (Bug 2a,
    # v0.0.36). The runner's lifetime lock is the real guard against two bridges.
    fd = acquire_lock(persona_dir)
    if fd is None:
        print(_LOCK_HELD_MSG, file=sys.stderr)
        return 2

    if needed:
        # The rebuild runs as its own process under this lock, before the bridge (S107). It comes before the stop intent is cleared
        # and the start back-off is reset below: a start that fails here never got going, so a kill's intent and the back-off stay.
        try:
            step_rc = _rebuild_under_lock(persona_dir)
        except BaseException:
            release_lock(persona_dir, fd)  # Ctrl-C in the chat REPL's in-process start must not leave the lock held
            raise
        if step_rc is not None:
            release_lock(persona_dir, fd)
            return step_rc

    client_origin = getattr(args, "client_origin", "cli")
    if client_origin == "launchd":
        try:
            from brain.service.launchd import truncate_launchd_logs_if_large

            truncate_launchd_logs_if_large(args.persona)
        except Exception:
            logger.debug("launchd log truncation skipped", exc_info=True)

    # #456 (D12, Q-B): no synchronous recovery here. The bridge's first idle pass saves every open session through
    # the gated job session_snapshot_prune (inside the background slot), whatever ended the last bridge.

    # #456 (D25, S6-f): an explicit start clears a kill's stop intent, only now (after the ALIVE refusal above, so a
    # launch check during an in-flight kill leaves it alone).
    from brain.bridge import shutdown_progress

    shutdown_progress.delete_record(persona_dir, shutdown_progress.INTENT_FILENAME)
    # D26: an explicit start does not wait and starts the start back-off over (a service run already waiting ends its
    # wait at once, and the service run this start causes is the first of a new series).
    start_backoff.reset(persona_dir)

    log_path = get_log_dir() / f"bridge-{persona_dir.name}.log"
    idle = float(args.idle_shutdown) * 60 if args.idle_shutdown > 0 else None
    _rotate_bridge_log(log_path)  # under the lock: see _rotate_bridge_log
    release_lock(persona_dir, fd)  # hand the lock to the child runner
    # #456 (D20, S-h): a persona whose service is installed starts through its manager, so the manager owns the bridge
    # (and a later stop goes the service way); the readiness wait gets what is left of the same 50 s budget.
    if _service_installed(args.persona):
        t0 = time.monotonic()
        try:
            started = _start_service(args.persona)
        except Exception as exc:  # noqa: BLE001 - reported like any failed start
            print(f"service start failed: {exc}", file=sys.stderr)
            return 1
        if started is not None and started.returncode != 0:
            # Review S12: the manager refused the start; say so now, not after the readiness wait.
            detail = (started.stderr or started.stdout or "").strip()
            print(f"service start failed (exit {started.returncode}): {detail}", file=sys.stderr)
            return 1
        return _await_service_readiness(persona_dir, _READINESS_BUDGET_S - (time.monotonic() - t0), out)
    pid = spawn_detached(persona_dir, idle, client_origin, log_path)
    try:
        return _await_readiness(persona_dir, pid, log_path, out)
    finally:
        _spawned_children.pop(pid, None)


# Readiness window: Windows cold-boot (persona load + soul review; before #456 also recovery) routinely exceeds 5s; the old 5s deadline
# killed a healthy-but-slow bridge. (Bug 2b, v0.0.36.)
# The wait can outlast it: a service run rebuilds (the rebuild child, S107) before it becomes the bridge, and ``cmd_start``'s wait
# after ``_start_service`` and ``cmd_restart``'s exit-2 wait are waits on such a run (the direct spawn shares the one helper, P1).
# The budget therefore pauses while an engine progress record names a live writer (``_ReadinessBudget``); with no record it is
# this plain 50 s. The Rust caller's wait for ``nell supervisor start`` (lib.rs ``wait_for_start``) pauses on the same record.
_READINESS_BUDGET_S = 50.0


class _ReadinessBudget:
    """A readiness wait's deadline, paused while an engine progress record names a live writer (Engine 6, HR5 part 4).

    ``expired()`` is asked once per 0.1 s poll. It reads the record through ``engine_files.read_engine_progress`` (liveness
    by ``holder_alive``; the reader never raises, and a missing or unreadable record counts as absent for that poll). A poll
    that finds a record pushes the deadline out by the time since the previous poll, so the clock stands still while the
    record shows; time before the record appears, and after it is gone, counts as ordinary budget. With no record ever it is
    exactly ``start + budget_s``.
    """

    def __init__(self, persona_dir: Path, budget_s: float) -> None:
        self._persona_dir = persona_dir
        self._last = time.time()
        self._deadline = self._last + max(0.0, budget_s)

    def expired(self) -> bool:
        from brain.emotion import engine_files

        now = time.time()
        if engine_files.read_engine_progress(self._persona_dir) is not None:
            self._deadline += now - self._last
        self._last = now
        return now >= self._deadline


def _service_installed(persona: str) -> bool:
    from brain.bridge import kill

    return kill.service_installed(persona)


def _start_service(persona: str) -> subprocess.CompletedProcess[str] | None:
    """The manager's start call; its result carries the manager's return code and message."""
    from brain.service import current_backend

    return current_backend().start_service(persona=persona)


def _await_service_readiness(persona_dir: Path, budget_s: float, out: dict | None) -> int:
    """Readiness of a bridge the service manager started (#456 D20): a record of a live bridge (by its token) whose
    /health answers, within what is left of the 50 s budget (which pauses while a rebuild's progress record shows)."""
    budget = _ReadinessBudget(persona_dir, budget_s)
    while not budget.expired():
        time.sleep(0.1)
        s = state_file.read(persona_dir)
        if s is None or not s.port or state_file.bridge_liveness(s, persona_dir) != state_file.ALIVE:
            continue
        try:
            headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
            r = httpx.get(
                f"http://127.0.0.1:{s.port}/health",
                headers=headers,
                timeout=1.0,
                trust_env=False,  # never through a proxy (#475)
            )
        except httpx.HTTPError:
            continue
        if r.status_code == 200:
            print(f"bridge started by the service on port {s.port} (pid {s.pid})")
            if out is not None:
                out["readiness"] = BridgeReadiness(pid=s.pid, port=s.port, auth_token=s.auth_token)
            return 0
    print("the service did not bring the bridge up in time; see `nell service status`", file=sys.stderr)
    return 1


def _await_readiness(persona_dir: Path, pid: int, log_path: Path, out: dict | None) -> int:
    """cmd_start's readiness wait for the child it just spawned (see cmd_start)."""
    proc = _spawned_children.get(pid)

    budget = _ReadinessBudget(persona_dir, _READINESS_BUDGET_S)
    while not budget.expired():
        time.sleep(0.1)
        s = state_file.read(persona_dir)
        if s is not None and s.pid == pid and s.port:
            try:
                headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
                r = httpx.get(
                    f"http://127.0.0.1:{s.port}/health",
                    headers=headers,
                    timeout=1.0,
                    trust_env=False,  # the loopback bridge is never reached through a proxy (#475)
                )
                if r.status_code == 200:
                    print(f"bridge started on port {s.port} (pid {pid})")
                    if out is not None:
                        out["readiness"] = BridgeReadiness(
                            pid=pid, port=s.port, auth_token=s.auth_token
                        )
                    return 0
            except httpx.HTTPError:
                pass
        # The child already exited without becoming ready: stop waiting now,
        # not at the deadline. poll() is None for a child still starting up,
        # so a slow start never trips this; with bridge_python() the Popen is
        # the runner itself (and a Windows venv redirector, were one in
        # between, stays alive exactly as long as the runner it wraps).
        rc = proc.poll() if proc is not None else None
        if rc is not None:
            if rc == 2:
                # runner.main's own refusal (S57): a live bridge is recorded,
                # or another starter's child holds the lock. Same wording as
                # cmd_start's own two refusal paths.
                if state_file.is_running(persona_dir):
                    return _refuse_already_running(persona_dir, out)
                print(_LOCK_HELD_MSG, file=sys.stderr)
                return 2
            print(
                f"bridge process (pid {pid}) exited with code {rc} before becoming ready. "
                f"Inspect log at {log_path}",
                file=sys.stderr,
            )
            return 1
    # Readiness failed — kill the orphan child and tell the user where to look.
    _kill_if_alive(pid, signal.SIGTERM)  # already-dead is fine either way
    print(
        f"bridge spawned (pid {pid}) but /health did not respond within its readiness budget (50s, not counting a rebuild in progress) — "
        f"killed orphan child. Inspect log at {log_path}",
        file=sys.stderr,
    )
    return 1


def _ended_on_a_plain_stop(st: state_file.BridgeState | None) -> bool:
    """The record says its bridge ended on a plain ``nell supervisor stop`` (user_stop): a person asked for no bridge.
    Not a restart's stop (app_restart; manual_restart, which ``nell supervisor restart`` records) nor a hard exit
    (it may end a restart): a start follows those, and a queued service run that skipped would leave the restart's
    own start facing a held lock or an exiting service job, so no bridge (maintainer's Q1). Not an idle stop (nobody
    asked for anything), not an OS signal (it cannot tell a user's kill from a logout or a service manager's stop),
    not a crash. A kill is the respawn guard's (D25), not this."""
    return st is not None and st.phase != "running" and st.stop_reason == REASON_USER_STOP


def cmd_run(args) -> int:
    """Run the bridge in the foreground for OS service managers.

    Unlike ``cmd_start``, this does not fork or detach. The current process is
    the bridge process until uvicorn exits, which is exactly what launchd wants
    to supervise. A per-persona lock is held for the lifetime of the process; a
    run that finds it held waits for it (D4). A service run that waited for a
    running bridge does not start when that bridge ended on a plain ``nell
    supervisor stop`` (see ``_ended_on_a_plain_stop``); after a restart's
    stop, a hard exit, an idle stop, an OS signal or a crash it starts.

    A service run that follows starts which never proved themselves first waits out the next step of the start
    back-off (``start_backoff``, D26), before the lock and without holding anything; a start it makes is noted under
    the lock, right before the bridge runs.

    With the lock held and the guard checks passed, a run for a persona that needs the rebuild runs it first
    (``_rebuild_under_lock``, S107); a step that fails returns its code, noting a start when guarded.
    """
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    client_origin = getattr(args, "client_origin", "cli")
    guarded = client_origin in respawn_guard.SERVICE_ORIGINS
    # #456 (D25): a service respawn after a user kill of this boot does not start (checked before the lock and again
    # right after acquiring it). An app / CLI run never applies the guard.
    if guarded and respawn_guard.guard_fires(persona_dir):
        print("not starting: stopped on request", file=sys.stderr)
        logger.info("not starting: stopped on request (user kill of this boot)")
        return 0
    if guarded:
        # D26: spaced restarts of a bridge that keeps failing to start. Before the lock (nothing held, no watcher file
        # written, so every other command sees no bridge and no starter) and interruptible: an explicit start deletes
        # the record, a user kill fires the guard. SIGTERM and Task Scheduler's end act at once (no handler yet).
        # The wait's log lines reach the runtime log through a handler that keeps no file open (a run that waits or is
        # queued must not block the running bridge's log rotation); the persistent handler comes after the lock.
        from brain.bridge import runner

        transient_log = runner._attach_transient_runtime_log(persona_dir)  # noqa: SLF001
        try:
            go_on = start_backoff.wait_before_start(
                persona_dir, interrupted=lambda: respawn_guard.guard_fires(persona_dir)
            )
        except KeyboardInterrupt:
            return 0
        finally:
            runner._detach_runtime_log(transient_log)  # noqa: SLF001
        if not go_on:
            print("not starting: stopped on request", file=sys.stderr)
            logger.info("not starting: stopped on request (user kill of this boot)")
            return 0

    # D4 (#177 S3): an OS service must not exit 2 here (systemd / launchd / Task Scheduler read it as a failure and
    # respawn forever). Whether a live bridge or a starter holds the lock, wait for it, then run.
    fd = acquire_lock(persona_dir)
    waited_for_a_bridge = False
    if fd is None:
        cur = state_file.read(persona_dir)
        if cur is not None and state_file.bridge_liveness(cur, persona_dir) == state_file.ALIVE:
            waited_for_a_bridge = True
            print(f"waiting for the running bridge (pid {cur.pid}) to exit", file=sys.stderr)
        else:
            print("waiting for the bridge lock (another start is in progress)", file=sys.stderr)
        fd = acquire_lock(persona_dir, blocking=True)
        if fd is None:
            print("bridge lock unavailable", file=sys.stderr)
            return 1

    try:
        if guarded:
            # The persistent runtime log, now that the lock is held (so the unit refresh below logs to it too;
            # run_bridge_foreground sets it up again later and adds nothing).
            from brain.bridge import runner

            runner._setup_runtime_logging(persona_dir)  # noqa: SLF001 - shared with run_bridge_foreground
        if guarded and respawn_guard.guard_fires(persona_dir):
            # Re-checked under the lock, BEFORE run_bridge_foreground deletes the watcher files (S6-a, S7-g).
            print("not starting: stopped on request", file=sys.stderr)
            logger.info("not starting: stopped on request (user kill of this boot)")
            return 0
        if guarded and waited_for_a_bridge and _ended_on_a_plain_stop(state_file.read(persona_dir)):
            # Pre-merge review O-2: a service run queued behind a running bridge does not undo a plain `nell
            # supervisor stop` (exit 0: success to every service manager, so nothing restarts it).
            print("not starting: the running bridge was stopped on request", file=sys.stderr)
            logger.info("not starting: the bridge this run waited for was stopped on request")
            return 0
        if _rebuild_needed(persona_dir, False):
            # The rebuild step (S107, HS6 (b)), the one a service start makes because it does not pass through cmd_start. A failed
            # step counts as a start the back-off must see: start_backoff counts only noted starts, and without this systemd's
            # RestartSec=2 would trip its start limit on a step that fails at once. The lock is released by the finally below.
            step_rc = _rebuild_under_lock(persona_dir)
            if step_rc is not None:
                if guarded:
                    start_backoff.note_start(persona_dir)
                return step_rc
        if client_origin == "systemd":
            try:
                from brain.service.systemd import refresh_installed_unit

                refresh_installed_unit(args.persona)
            except Exception:  # noqa: BLE001 - best effort; the unit keeps working without the line
                logger.warning("systemd unit refresh failed", exc_info=True)
        if client_origin == "launchd":
            try:
                from brain.service.launchd import truncate_launchd_logs_if_large

                truncate_launchd_logs_if_large(args.persona)
            except Exception:
                logger.debug("launchd log truncation skipped", exc_info=True)

        # #456 (D12): no synchronous recovery; the first idle pass saves the sessions (see cmd_start).
        from brain.bridge.runner import run_bridge_foreground

        if guarded:
            start_backoff.note_start(persona_dir)  # D26: stays until a recorded stop; SuccessWatch restarts the ladder

        idle = float(args.idle_shutdown) * 60 if args.idle_shutdown > 0 else None
        return run_bridge_foreground(
            persona_dir,
            client_origin=client_origin,
            idle_shutdown_seconds=idle,
        )
    finally:
        release_lock(persona_dir, fd)


def _request_shutdown_via_http(
    s: state_file.BridgeState, *, timeout: float = 3.0, reason: str | None = None
) -> httpx.Response:
    """POST /supervisor/shutdown. Returns the response (202 = the stop began; 503 ``bridge_stopping`` = it had begun
    already); raises on a transport failure (the listener closes once a stop is under way)."""
    if s.port is None:
        raise RuntimeError("bridge state missing port")
    headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
    return httpx.post(
        f"http://127.0.0.1:{s.port}/supervisor/shutdown",
        headers=headers,
        json={"reason": reason} if reason else None,
        timeout=timeout,
        trust_env=False,  # never through a proxy (#475)
    )


# A pre-PR bridge (no identity token, proven by its command line) gets the old request and this long to go before the
# kill helper ends it with its tree (D22). The value is the app's existing request budget before its kill
# (TIMEOUT_SHUTDOWN_MS, app/src/hooks/useRestartBridge.ts), so a pre-PR bridge is treated exactly as the app treats it.
_PRE_PR_REQUEST_BUDGET_S = 3.0


def _stop_began(s: state_file.BridgeState, persona_dir: Path, response: httpx.Response | None) -> bool:
    from brain import process_identity
    from brain.bridge import shutdown_progress

    if response is not None:
        if response.status_code == 202:
            return True
        try:
            if response.status_code == 503 and response.json().get("code") == "bridge_stopping":
                return True
        except ValueError:
            pass
    rec = shutdown_progress.read_record(persona_dir, shutdown_progress.PROGRESS_FILENAME)
    if process_identity.names_this_bridge(rec, s.pid, s.pid_token):
        return True
    cur = state_file.read(persona_dir)
    return bool(cur is not None and cur.pid == s.pid and cur.phase == "stopping")


def _print_kill_result(result) -> int:  # noqa: ANN001
    print(result.message)
    return 0 if result.succeeded else 1


def cmd_stop(args, *, reason: str | None = REASON_USER_STOP, kill_pre_pr: bool = True) -> int:
    """`nell supervisor stop` (#456 Q-A, D16, D22): ask the bridge to stop, then watch until it is gone.

    ``reason`` is what the stop request tells the bridge to record: ``"user_stop"`` for a plain stop, ``None`` for a
    restart's stop (recorded as ``manual_restart``). ``kill_pre_pr`` False (the start's stop before a rebuild, S110: no
    automatic hard kill) leaves a pre-PR bridge that does not stop in time running: the stop reports it and returns 1
    instead of calling the kill helper; stop and restart keep the default (True).

    ``--kill``: the kill helper (a bridge run by an OS service first gets the hard-exit request and up to
    ``--timeout`` to go, then its service manager's stop; any other bridge is killed at once). A kill that leaves a
    process running exits 1 (``KillResult.succeeded``). A pre-PR bridge (no identity token): the old request, then
    the kill helper after a few seconds. A stale record: "not running". If the request fails and no stop is recorded:
    POSIX sends SIGTERM when the identity holds; Windows refuses unless ``--force`` (then the kill helper).
    """
    from brain.bridge import kill as kill_mod
    from brain.bridge import stop_watch
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    timeout = float(getattr(args, "timeout", stop_watch.HARD_EXIT_OFFER_AFTER_S))
    s = state_file.read(persona_dir)
    if s is None or s.pid is None:
        print("bridge not running")
        return 0
    live = state_file.bridge_liveness(s, persona_dir)
    if live == state_file.GONE:
        # A stale record (dead pid, a reused pid, or a pre-PR pid that is not a bridge for this persona): never signal.
        print("bridge not running (stale record)")
        _clear_stale_record(persona_dir, s)
        return 0
    if getattr(args, "kill", False):
        if live != state_file.ALIVE:
            print("bridge identity could not be confirmed; not killing", file=sys.stderr)
            return 1
        return _print_kill_result(
            kill_mod.kill_bridge(
                persona_dir,
                timeout=timeout,
                skip_hard_exit_request=bool(getattr(args, "skip_hard_exit_request", False)),
                expect_pid=getattr(args, "expect_pid", None),
                expect_token=getattr(args, "expect_token", None),
            )
        )
    if state_file.is_legacy(s):
        return _stop_pre_pr(persona_dir, s, timeout, kill_pre_pr=kill_pre_pr)

    response = None
    try:
        response = _request_shutdown_via_http(s, reason=reason)
    except Exception as exc:  # noqa: BLE001 - a closed listener: the stop may have begun already (attach)
        failure = exc
    else:
        failure = None
    if not _stop_began(s, persona_dir, response):
        if live != state_file.ALIVE:
            print(f"bridge identity could not be confirmed; not signalling pid {s.pid}: {failure}", file=sys.stderr)
            return 1
        if os.name == "nt":
            if getattr(args, "force", False):
                print("WARNING: forcing termination of a bridge that does not answer.", file=sys.stderr)
                return _print_kill_result(kill_mod.kill_bridge(persona_dir, timeout=timeout))
            print(
                f"bridge shutdown endpoint unreachable; not killing it: {failure}\n"
                "If the bridge is wedged, re-run with --force (or --kill) to end it.",
                file=sys.stderr,
            )
            return 1
        logger.warning("shutdown endpoint failed; falling back to SIGTERM on POSIX: %s", failure)
        if not _kill_if_alive(s.pid, signal.SIGTERM):
            print("bridge not running")
            return 0
    return stop_watch.watch_stop(
        persona_dir,
        pid=s.pid,
        token=s.pid_token,
        timeout=timeout,
        interactive=stop_watch._interactive(),
        now=stop_watch._clock.now,
        sleep=stop_watch._clock.sleep,
        prompt=stop_watch._prompt,
    )


def _stop_pre_pr(persona_dir: Path, s: state_file.BridgeState, timeout: float, *, kill_pre_pr: bool = True) -> int:
    """A proven pre-PR bridge (D22): the old request (skipped when its listener is closed), up to the app's budget,
    then the kill helper (with its tree). No display, no offer. With ``kill_pre_pr`` False the kill helper is not called: a
    bridge still running after the budget is reported and the result is 1."""
    from brain.bridge import kill as kill_mod

    try:
        _request_shutdown_via_http(s)
    except Exception:  # noqa: BLE001 - listener already closed: go straight to the wait
        pass
    deadline = time.time() + _PRE_PR_REQUEST_BUDGET_S
    while time.time() < deadline:
        if state_file.bridge_liveness(s, persona_dir) == state_file.GONE:
            break
        time.sleep(0.2)
    if state_file.bridge_liveness(s, persona_dir) == state_file.ALIVE:
        if not kill_pre_pr:
            print("The previous version of the bridge did not stop.", file=sys.stderr)
            return 1
        result = kill_mod.kill_bridge(persona_dir, timeout=timeout)
        if not result.succeeded:
            print(result.message, file=sys.stderr)
            return 1
    print("Stopped the previous version of the bridge. The next start will pick up anything left unfinished.")
    return 0


def _stop_before_service_change(persona: str, timeout: float) -> bool:
    """Stop a running bridge before a service uninstall / install (C54, C58 d): the shutdown request and the watcher.
    True once it has stopped; False when it is still stopping (no terminal, after ``timeout``)."""
    rc = cmd_stop(SimpleNamespace(persona=persona, timeout=timeout, force=False, kill=False,
                                  skip_hard_exit_request=False))
    from brain.paths import get_persona_dir

    return rc == 0 and state_file.liveness(get_persona_dir(persona)) != state_file.ALIVE


def _clear_stale_record(persona_dir: Path, s: state_file.BridgeState) -> None:
    """Clear a stale record's pid/port, only while holding the bridge lock (so no bridge is running or starting)."""
    fd = acquire_lock(persona_dir)
    if fd is None:
        return
    try:
        cur = state_file.read(persona_dir)
        if cur is not None and cur.pid == s.pid:
            cur.pid = None
            cur.port = None
            state_file.write(persona_dir, cur)
    except Exception:  # noqa: BLE001 - best effort; the record stays stale, which readers already handle
        logger.warning("could not clear the stale bridge record", exc_info=True)
    finally:
        release_lock(persona_dir, fd)


def status_dict(persona_dir: Path) -> dict:
    """The machine-readable bridge status (`nell supervisor status --json`): files and process checks only."""
    from brain.emotion import engine_files

    s = state_file.read(persona_dir)
    live = state_file.bridge_liveness(s, persona_dir) if s is not None else state_file.GONE
    proven = bool(
        s is not None
        and s.pid is not None
        and not s.pid_token
        and state_file.prove_bridge_cmdline(s.pid, persona_dir, s.persona or persona_dir.name)
    )
    return {
        "alive": live,
        "legacy": state_file.is_legacy(s),
        "proven": proven,
        "pid": s.pid if s else None,
        "pid_token": s.pid_token if s else None,
        "identity": s.identity if s else None,
        "phase": s.phase if s else None,
        "stop_reason": s.stop_reason if s else None,
        "stop_requested_at": s.stop_requested_at if s else None,
        "client_origin": s.client_origin if s else None,
        "recovery_needed": state_file.recovery_needed(persona_dir),
        # #456 (plan I9, S7-g, C48): the stop intent with its state, what is still running in a stop of THIS bridge,
        # the service install state, the supervisor's state and the kill helper's stated bound.
        "stop_intent": _status_intent(persona_dir),
        "remaining": _status_remaining(persona_dir, s),
        # HR5 of #633 (owner S101, S102): the start-up step in progress (the x10 fix or the first rebuild), only while the process
        # that writes it is alive; None otherwise. Written by `brain.emotion.engine_files.EngineProgress`.
        "engine_progress": engine_files.read_engine_progress(persona_dir),
        "service_installed": _status_service_installed(s.persona if s and s.persona else persona_dir.name),
        "supervisor": s.supervisor if s else None,
        "kill_bound_s": _kill_bound(),
    }


def _status_intent(persona_dir: Path) -> dict | None:
    from brain.bridge import kill

    return kill.read_intent(persona_dir)


def _status_remaining(persona_dir: Path, s: state_file.BridgeState | None) -> dict | None:
    from brain import process_identity
    from brain.bridge import shutdown_progress

    rec = shutdown_progress.read_record(persona_dir, shutdown_progress.PROGRESS_FILENAME)
    if s is None or not process_identity.names_this_bridge(rec, s.pid, s.pid_token):
        return None
    return {"count": rec.get("count"), "tasks": rec.get("tasks"), "phase": rec.get("phase"),
            "requested_at": rec.get("requested_at"), "reason": rec.get("reason"), "updated_at": rec.get("updated_at")}


def _status_service_installed(persona: str) -> bool:
    from brain.bridge import kill

    try:
        return kill.service_installed(persona)
    except Exception:  # noqa: BLE001
        return False


def _kill_bound() -> float:
    from brain.bridge import kill

    return kill.kill_bound_s()


def cmd_status(args) -> int:
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None:
        print("bridge: not running (no state file)")
        return 0
    if state_file.is_running(persona_dir):
        try:
            headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
            r = httpx.get(
                f"http://127.0.0.1:{s.port}/health",
                headers=headers,
                timeout=1.0,
                trust_env=False,  # never through a proxy (#475)
            )
            health = r.json()
            print(f"bridge: running pid={s.pid} port={s.port}")
            print(f"  uptime_s: {health['uptime_s']}")
            print(f"  sessions_active: {health['sessions_active']}")
            print(f"  supervisor: {health['supervisor_thread']}")
            print(f"  pending_alarms: {health['pending_alarms']}")
        except httpx.HTTPError as e:
            print(f"bridge: pid {s.pid} alive but /health unreachable: {e}", file=sys.stderr)
            return 1
    elif state_file.recovery_needed(persona_dir):
        pid_note = f" (pid {s.pid})" if s.pid is not None else ""
        print(f"bridge: previous process crashed dirty{pid_note}. The next start will recover.")
    else:
        print(f"bridge: stopped cleanly at {s.stopped_at}")
    return 0


def cmd_tail(args) -> int:
    """Subscribe to /events and print every event as a JSON line.

    Auth via Sec-WebSocket-Protocol: bearer, <token> — the only auth path
    the server accepts. Previously this used a ?token= query string, which
    (a) the server doesn't read, so tail was silently broken in any
    config with auth enabled, and (b) leaks the bearer token through
    process listings and proxy logs.
    """
    from brain.bridge.loopback import direct_ws_connect as connect  # never through a proxy (#475)
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None or not state_file.is_running(persona_dir):
        print("bridge not running", file=sys.stderr)
        return 1
    url = f"ws://127.0.0.1:{s.port}/events"
    subprotocols = ["bearer", s.auth_token] if s.auth_token else None
    try:
        with connect(url, subprotocols=subprotocols) as ws:
            while True:
                msg = ws.recv()
                print(msg)
    except KeyboardInterrupt:
        return 0


def cmd_restart(args) -> int:
    """Stop the bridge, then start it again, only after the old bridge is gone (#456 C16 h).

    ``cmd_stop`` returns 0 only once the old bridge (its pid + token) is gone, or when none was running; restart then
    starts. Any other result (still stopping without a terminal, a refused stop) aborts the restart with that code. A
    start that finds another one already under way (exit 2) waits for that bridge to answer.
    """
    print("stopping bridge...")
    # No reason: the bridge records "manual_restart", so a service run queued behind it starts (see
    # _ended_on_a_plain_stop); a plain stop records "user_stop".
    stop_rc = cmd_stop(args, reason=None)
    if stop_rc != 0:
        print(f"restart aborted: stop failed (exit {stop_rc})", file=sys.stderr)
        return stop_rc
    print("starting bridge...")
    rc = cmd_start(args)
    if rc == 2:
        # A start is already under way: a service run queued behind the old bridge woke at its death and holds the
        # lock, so its bridge is coming up. Wait for that bridge, as the app does (exit 2 is its "start under way").
        from brain.paths import get_persona_dir

        return _await_service_readiness(get_persona_dir(args.persona), _READINESS_BUDGET_S, None)
    return rc


# Test seam: setting this Event from a test allows the follow-mode loop
# to exit cleanly without raising KeyboardInterrupt. Production exit
# from follow mode is handled by the KeyboardInterrupt catch in
# cmd_tail_log itself.
_follow_should_stop: threading.Event | None = None


def cmd_tail_log(args) -> int:
    """Print the last N lines of the runtime log; -f to follow.

    Cross-platform: pure Python loop, no shell `tail` (Windows CI lacks it).
    Follow mode polls every 200ms. KeyboardInterrupt is treated as a clean
    exit (returns 0). Tests can interrupt the loop by setting the module-level
    `_follow_should_stop` Event before the call.
    """
    from brain.paths import get_persona_dir, runtime_log_path

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    # The runtime log, where logging goes (#558); bridge-<persona>.log only has the child's raw stdout/stderr.
    log_path = runtime_log_path(persona_dir.name)
    if not log_path.exists():
        print(
            f"bridge log not found at {log_path} — has the supervisor ever started?",
            file=sys.stderr,
        )
        return 1

    n = max(0, int(args.lines))
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            # deque(maxlen=n) keeps only the last n lines in memory as it
            # iterates the file — bounded regardless of log size, unlike
            # f.readlines() which materialised the whole file before slicing.
            if n > 0:
                for line in deque(f, maxlen=n):
                    print(line, end="")
            if not getattr(args, "follow", False):
                return 0
            st = os.fstat(f.fileno())
        # Follow mode polls by path and never holds the file open between polls: the runtime log rolls over by
        # rename (RotatingFileHandler), so a kept handle would go on reading the renamed file, and on Windows it
        # would make that rename fail.
        ident, pos = st.st_ino, st.st_size
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        stop = _follow_should_stop or threading.Event()
        try:
            while not stop.is_set():
                try:
                    st = log_path.stat()
                except FileNotFoundError:  # between the rename and the new file
                    st = None
                if st is not None and (st.st_ino != ident or st.st_size < pos):  # rolled over or truncated
                    ident, pos = st.st_ino, 0
                    decoder.reset()
                if st is not None and st.st_size > pos:
                    with log_path.open("rb") as fb:
                        fb.seek(pos)
                        chunk = fb.read()
                    pos += len(chunk)
                    print(decoder.decode(chunk), end="")
                else:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            return 0
        return 0
    except OSError as e:
        print(f"error reading {log_path}: {e}", file=sys.stderr)
        return 1
