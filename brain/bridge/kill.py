"""Kill a bridge and its whole process tree: the last-resort stop (#456 D17, D20, D22, D25; plan I5).

Used by ``nell supervisor stop --kill`` (and ``--skip-hard-exit-request``), the terminal watcher's escalation, the
Windows ``--force`` path and, through ``--kill``, the desktop app's forced path. A graceful stop never comes here.

The sequence (plan I5 steps 1-7):

1. Identity: the bridge must be ALIVE by its token (or, for a record written before the token existed, by its proven
   command line). Anything else: "not running" (GONE) or "identity could not be confirmed" (UNKNOWN); nothing is
   written and nothing is killed. The same ``psutil.Process`` object is used from here to the kill.
2. Signal handlers (main thread only; SIGHUP only where it exists) are installed before the stop intent is written
   ``pending`` with this helper's nonce; everything after runs in one ``try/finally``.
3. A read-only enumeration of the descendants (the "remembered" set).
4. Service ownership, asked only when the persona's service is installed.
5. A service-owned bridge first gets the hard-exit request (unless skipped) and up to ``timeout`` to go; if it does not,
   the service manager is asked to stop it (15 s cap), with the descendants enumerated again just before and after.
6. Commit (compare-and-set on the nonce, and only when no other live bridge has appeared for the persona) immediately
   before the first suspend or kill; then freeze the bridge and its tree, kill the bridge first, then the frozen tree.
7. Poll "gone" over the bridge and everything remembered for ``timeout``; kill any survivor once more; report.

The ``finally``: while the bridge's kill call has not been started, resume everything suspended, delete the intent
(only if it still holds this nonce) and withdraw the hard-exit request this call wrote (only if it names this bridge),
so an interrupted kill leaves nothing that could still end the bridge; once it has been started (the flag is set
immediately before the call, so an interrupt inside the call counts), kill everything frozen, the bridge included, and the
remembered descendants and keep the committed intent (the respawn guard honours it). Stop signals are ignored while
the ``finally`` runs.

``kill_command`` ends one running command (a CLI process the persona's start must not run beside) by the same freeze, kill and
poll (``_end_tree``, ``_release_tree``), with no intent, request or service step.
"""

from __future__ import annotations

import logging
import os
import secrets
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge import shutdown_progress, state_file

logger = logging.getLogger(__name__)

POLL_S = 0.2  # the "gone" poll step (daemon.cmd_stop's own poll)
MANAGER_CAP_S = 15.0  # one service-manager call
DEFAULT_TIMEOUT_S = 180.0  # the hard-exit offer threshold (spec section 9)


def kill_bound_s(timeout: float = DEFAULT_TIMEOUT_S, *, skip_hard_exit_request: bool = False) -> float:
    """The stated worst case of ``kill_bridge`` (S5-a): the hard-exit wait (unless skipped) + two manager calls + the
    kill poll."""
    return (0.0 if skip_hard_exit_request else timeout) + 2 * MANAGER_CAP_S + timeout


class _KillInterrupted(BaseException):
    """A stop signal delivered to the helper itself."""


@dataclass
class KillResult:
    outcome: str  # "not_running" | "unknown_identity" | "other_bridge" | "killed"
    message: str
    survivors: list[int] = field(default_factory=list)
    committed: bool = False

    @property
    def succeeded(self) -> bool:
        """The one success rule every caller uses (``cmd_stop``, the pre-PR stop, the terminal watcher): no bridge
        is left and nothing it started is still running. A caller that goes on to start a new bridge on success must
        never do so beside a survivor (review of #458)."""
        return self.outcome in ("killed", "not_running") and not self.survivors


# ---------------------------------------------------------------- the stop intent (D6 c, D25)


def _intent_path(persona_dir: Path) -> Path:
    return shutdown_progress.path(persona_dir, shutdown_progress.INTENT_FILENAME)


def read_intent(persona_dir: Path) -> dict | None:
    return shutdown_progress.read_record(persona_dir, shutdown_progress.INTENT_FILENAME)


def _write_intent(persona_dir: Path, intent: dict) -> None:
    shutdown_progress.write_record(persona_dir, shutdown_progress.INTENT_FILENAME, intent)


def _cas_intent(persona_dir: Path, nonce: str, update: Callable[[dict], dict | None]) -> bool:
    """Compare-and-set on the intent: apply ``update`` (None = delete) only when the file still holds ``nonce``."""
    with _intent_lock:
        cur = read_intent(persona_dir)
        if not isinstance(cur, dict) or cur.get("nonce") != nonce:
            return False
        new = update(cur)
        if new is None:
            shutdown_progress.delete_record(persona_dir, shutdown_progress.INTENT_FILENAME)
        else:
            _write_intent(persona_dir, new)
        return True


_intent_lock = threading.Lock()


# ---------------------------------------------------------------- service seams (I5 steps 4, 5; tests spy these)


def _platform() -> str:
    """Seam for tests: patching ``sys.platform`` globally reaches the stdlib and process-wide caches (#662, #710)."""
    return sys.platform


def _service_backend(platform: str | None = None):
    platform = platform or _platform()
    try:
        if platform == "darwin":
            from brain.service import launchd as backend
        elif platform == "linux":
            from brain.service import systemd as backend
        elif platform.startswith("win"):
            from brain.service import windows_service as backend
        else:
            return None
        return backend
    except Exception:  # noqa: BLE001 - no backend on this platform: never service-owned
        return None


def service_installed(persona: str, platform: str | None = None) -> bool:
    """Is the persona's OS service installed? (The seam tests stub; the check itself is below.)"""
    return _service_installed_check(persona, platform)


def _service_installed_check(persona: str, platform: str | None = None) -> bool:
    platform = platform or _platform()
    backend = _service_backend(platform)
    if backend is None:
        return False
    try:
        if platform.startswith("win"):
            return bool(backend.service_status(persona=persona).installed)
        paths = backend.paths_for_persona(persona)
        return Path(getattr(paths, "unit_path", None) or paths.plist_path).exists()
    except Exception:  # noqa: BLE001 - an unreadable install state is "not installed": no manager call
        return False


def owned_by_service(persona: str, pid: int, client_origin: str | None, platform: str | None = None) -> bool:
    """Asked only when the service is installed (S6-c): does the manager say this pid is its job?"""
    platform = platform or _platform()
    backend = _service_backend(platform)
    if backend is None:
        return False
    try:
        if platform == "linux":
            r = backend.run_systemctl(["show", "-p", "MainPID", "--value", backend.service_unit_name(persona)])
            return r.returncode == 0 and r.stdout.strip() == str(pid)
        if platform == "darwin":
            r = backend.run_launchctl(["print", backend.launchctl_target(persona)], timeout=MANAGER_CAP_S)
            lines = [ln.strip() for ln in (r.stdout or "").splitlines()]
            return r.returncode == 0 and lines.count(f"pid = {pid}") == 1
        if platform.startswith("win"):
            return client_origin == "task-scheduler"
    except Exception:  # noqa: BLE001 - an unanswerable question is "not owned"
        return False
    return False


def manager_stop(persona: str, platform: str | None = None) -> None:
    """The manager's stop for an owned bridge that did not end on the hard-exit request (15 s cap; any result goes on
    to the helper's own kill)."""
    platform = platform or _platform()
    backend = _service_backend(platform)
    if backend is None:
        return
    try:
        if platform == "linux":
            backend.run_systemctl(["stop", "--no-block", backend.service_unit_name(persona)])
        elif platform == "darwin":
            backend.run_launchctl(["bootout", backend.launchctl_target(persona)], timeout=MANAGER_CAP_S)
        # Windows: no manager call; the helper's own kill (exit code 0) ends the task's process.
    except Exception:  # noqa: BLE001 - a failed or timed-out manager call: the helper kills it itself
        logger.warning("service manager stop failed; killing the bridge directly", exc_info=True)


# ---------------------------------------------------------------- process helpers


def _children(proc) -> list:
    try:
        return proc.children(recursive=True)
    except Exception:  # noqa: BLE001 - the bridge is gone: no new members
        return []


def _alive(proc) -> bool:
    try:
        import psutil

        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def _kill_proc(proc) -> None:
    try:
        if _platform().startswith("win"):  # pragma: no cover - Windows CI leg
            _terminate_exit_0(proc)
        else:
            proc.kill()
    except Exception:  # noqa: BLE001 - gone already
        pass


def _kernel32():  # pragma: no cover - Windows CI leg
    import ctypes

    return ctypes.windll.kernel32  # type: ignore[attr-defined]


def _terminate_exit_0(proc) -> None:  # noqa: ANN001
    """Windows: end the process with exit code 0, so Task Scheduler does not treat the end as a failure.

    The open handle pins the pid, so once ``proc.is_running()`` (psutil's create-time check) confirms the pid is
    still the remembered process, the handle cannot reach another one (review of #458: a bare pid could)."""
    k32 = _kernel32()
    handle = k32.OpenProcess(0x0001, False, proc.pid)  # PROCESS_TERMINATE
    if handle:
        try:
            if proc.is_running():
                k32.TerminateProcess(handle, 0)
        finally:
            k32.CloseHandle(handle)


def _helper_may_run(intent: dict) -> bool:
    """May the helper that committed ``intent`` still be running? (review S3) A helper killed after its commit leaves
    the intent behind; once that helper is gone the intent no longer blocks a new kill. An intent that names no
    helper (or one whose identity cannot be read) counts as running until it is older than the helper's stated bound."""
    from brain import process_identity

    hp, ht = intent.get("helper_pid"), intent.get("helper_token")
    if isinstance(hp, int) and isinstance(ht, str) and ht:
        live = process_identity.matches(hp, ht)
        if live == process_identity.GONE:
            return False
        if live == "alive":
            return True
    try:
        at = datetime.fromisoformat(str(intent.get("at"))).timestamp()
    except ValueError:
        return True
    return time.time() - at <= kill_bound_s(DEFAULT_TIMEOUT_S)


def _other_live_bridge(persona_dir: Path, target_pid: int) -> bool:
    """Has another bridge started for this persona meanwhile? (C45 i: the bridge lock is held by a pid other than the
    target's, or the record names another ALIVE pid)."""
    import psutil

    from brain.bridge.daemon import lock_holder

    holder = lock_holder(persona_dir)
    if holder is not None and holder != target_pid and psutil.pid_exists(holder):
        return True
    st = state_file.read(persona_dir)
    if st is None or st.pid is None or st.pid == target_pid:
        return False
    return state_file.bridge_liveness(st, persona_dir) == state_file.ALIVE


# ---------------------------------------------------------------- the tree (shared by kill_bridge and kill_command)


def _end_tree(
    root,  # noqa: ANN001 - a psutil.Process
    suspended: list,
    remembered: dict[int, object],
    timeout: float,
    *,
    hook: Callable[[str], None] | None = None,
    on_root_killed: Callable[[], None] | None = None,
) -> list[int]:
    """Freeze ``root`` and its descendants, kill the root first, then the frozen tree, poll for ``timeout`` and return the
    pids still alive after a second kill. Plan I5 steps 6 (from the freeze) and 7.

    ``suspended`` and ``remembered`` belong to the caller and are filled in place, so that its ``finally`` sees what was
    frozen even when this call is interrupted; ``_release_tree`` is the one place that resumes or finishes the kill. This
    function never resumes. ``hook`` is called with a step name (``after_bridge_suspend``, ``before_bridge_kill``,
    ``after_bridge_kill``); ``on_root_killed`` is called immediately before the root's kill call (so a caller interrupted inside
    that call already reads "root killed" and ``_release_tree`` ends the tree instead of resuming it), before the hook
    ``after_bridge_kill`` follows the call, and also when the root was already gone at the suspend.
    """
    import psutil

    def _hook(name: str) -> None:
        if hook is not None:
            hook(name)

    def _root_killed() -> None:
        if on_root_killed is not None:
            on_root_killed()

    frozen = False
    if _alive(root):
        try:
            root.suspend()
            frozen = True
        except psutil.NoSuchProcess:
            pass  # it ended between the check and the suspend (review S4): the "already gone" branch below
    if frozen:
        suspended.append(root)
        _hook("after_bridge_suspend")
        # Every pid tried is remembered, suspended or not (review M3): a descendant whose suspend() fails and
        # that stays listed cannot keep this loop going; it is killed with the tree below.
        tried = {root.pid}
        while True:
            new = [p for p in _children(root) if p.pid not in tried]
            if not new:
                break
            for p in new:
                tried.add(p.pid)
                try:
                    p.suspend()
                    suspended.append(p)
                except psutil.Error:
                    pass
                remembered.setdefault(p.pid, p)
        _hook("before_bridge_kill")
        _root_killed()  # before the call, so an interrupt inside it already reads "root killed" (``_release_tree`` then ends the tree)
        _kill_proc(root)
        _hook("after_bridge_kill")
        for p in suspended[1:]:
            _kill_proc(p)
    else:
        _root_killed()
    for p in remembered.values():
        if _alive(p):
            _kill_proc(p)

    # Step 7: poll gone over everything.
    everyone = [root, *remembered.values()]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(_alive(p) for p in everyone):
        time.sleep(POLL_S)
    survivors = [p.pid for p in everyone if _alive(p)]
    for p in everyone:
        if _alive(p):
            _kill_proc(p)
    return survivors


def _release_tree(suspended: list, remembered: dict[int, object], root_kill_started: bool) -> None:
    """The one place a frozen tree is let go. The root's kill has not been started (``on_root_killed`` has not run): resume
    everything suspended. It has: finish the kill of everything frozen and the remembered ones still alive. The root is
    ``suspended[0]`` whenever it was frozen, so it is killed here too: ``_kill_proc`` swallows every exception, so a root already
    gone is a no-op, and an interrupt before its kill call acted still ends it with its descendants instead of leaving it frozen."""
    if not root_kill_started:
        for p in suspended:
            try:
                p.resume()
            except Exception:  # noqa: BLE001
                pass
        return
    for p in [*suspended, *remembered.values()]:
        if _alive(p):
            _kill_proc(p)


# ---------------------------------------------------------------- the helper


def kill_bridge(
    persona_dir: Path,
    persona: str | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    skip_hard_exit_request: bool = False,
    expect_pid: int | None = None,
    expect_token: str | None = None,
    _hooks: dict | None = None,
) -> KillResult:
    """Kill the persona's bridge and its tree (module docstring). ``_hooks`` are test seams (named step callbacks).

    ``expect_pid`` + ``expect_token`` (the app's watched bridge; review S8, D16 amended): when given, the record must
    name exactly that bridge, or nothing is written or killed (outcome ``bridge_changed``)."""
    import psutil

    from brain import process_identity

    hooks = _hooks or {}
    persona_dir = Path(persona_dir)
    persona = persona or persona_dir.name
    st = state_file.read(persona_dir)
    if expect_pid is not None and (st is None or st.pid != expect_pid or st.pid_token != expect_token):
        return KillResult("bridge_changed", "bridge_changed: another bridge is running now; nothing killed")
    live = state_file.bridge_liveness(st, persona_dir)
    if live == state_file.UNKNOWN:
        return KillResult("unknown_identity", "bridge identity could not be confirmed; not killing")
    if live != state_file.ALIVE or st is None or st.pid is None:
        return KillResult("not_running", "bridge not running")
    try:
        bridge = psutil.Process(st.pid)
    except psutil.Error:
        return KillResult("not_running", "bridge not running")
    if st.pid_token and process_identity.token_of(st.pid) != st.pid_token:
        return KillResult("not_running", "bridge not running")
    legacy = state_file.is_legacy(st)
    cur_intent = read_intent(persona_dir)
    if (
        isinstance(cur_intent, dict)
        and cur_intent.get("state") == "committed"
        and cur_intent.get("pid") == st.pid
        and cur_intent.get("pid_token") == st.pid_token
        and _helper_may_run(cur_intent)
    ):
        # C45 (i): another helper has committed to killing this bridge; never downgrade or delete its intent.
        return KillResult("in_progress", "another kill of this bridge is in progress")

    # Step 2: handlers first, then the pending intent, both inside the try (review S2): a signal in between still
    # reaches the finally, which deletes our intent (a no-op before it was written) and restores the handlers.
    prev: dict[int, object] = {}
    on_main = threading.current_thread() is threading.main_thread()
    fired: list[int] = []

    def _raise(signum, frame):  # noqa: ANN001
        # One-shot (review S2): only the first stop signal interrupts; a second one, arriving before the finally
        # has masked signals, must not skip the resume.
        if fired:
            return
        fired.append(signum)
        raise _KillInterrupted(signum)

    # Ctrl-C too (review R2-S2): every stop signal gets the one-shot handler.
    stop_signals = [signal.SIGTERM, signal.SIGINT] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
    nonce = secrets.token_hex(8)

    remembered: dict[int, object] = {}
    suspended: list = []
    root_kill_started = False  # set by ``_end_tree`` immediately before the bridge's kill call (also when the root was already gone)
    request_written = False

    def _remember(procs) -> None:
        for p in procs:
            remembered.setdefault(p.pid, p)

    def _hook(name: str) -> None:
        cb = hooks.get(name)
        if cb is not None:
            cb()

    try:
        if on_main:
            for sig in stop_signals:
                prev[sig] = signal.signal(sig, _raise)
        intent = shutdown_progress.stop_intent(
            pid=st.pid, pid_token=st.pid_token, state="pending", nonce=nonce, at=datetime.now(UTC).isoformat(),
            boot=process_identity.boot_id(), legacy=legacy, helper_pid=os.getpid(),
            helper_token=process_identity.own_token(),
        )
        _write_intent(persona_dir, intent)

        # Step 3: read-only enumeration.
        _remember(_children(bridge))
        _hook("after_enumeration")

        # Steps 4-5: service ownership; an owned bridge gets the hard-exit request, then the manager's stop.
        if service_installed(persona) and owned_by_service(persona, st.pid, st.client_origin):
            if not legacy and not skip_hard_exit_request:
                shutdown_progress.write_record(
                    persona_dir,
                    shutdown_progress.REQUEST_FILENAME,
                    shutdown_progress.hard_exit_request(
                        pid=st.pid, pid_token=st.pid_token, requested_at=datetime.now(UTC).isoformat()
                    ),
                )
                request_written = True
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and _alive(bridge):
                    _hook("hard_exit_wait")
                    time.sleep(POLL_S)
            if _alive(bridge):
                _remember(_children(bridge))
                _hook("before_manager")
                manager_stop(persona)
                if _alive(bridge):
                    _remember(_children(bridge))

        # Step 6: commit, then freeze and kill.
        def _commit(cur: dict) -> dict | None:
            if _other_live_bridge(persona_dir, st.pid):
                return None
            return {**cur, "state": "committed"}

        if not _cas_intent(persona_dir, nonce, _commit) or read_intent(persona_dir) is None:
            # Our pending intent is gone (another helper's file, or another bridge appeared: deleted above).
            _cas_intent(persona_dir, nonce, lambda _cur: None)
            return KillResult("other_bridge", "another bridge is now running; nothing killed")
        _hook("committed")

        def _root_killed() -> None:
            nonlocal root_kill_started
            root_kill_started = True

        survivors = _end_tree(bridge, suspended, remembered, timeout, hook=_hook, on_root_killed=_root_killed)
        if survivors:
            return KillResult("killed", f"processes still running after the kill: {survivors}", survivors, True)
        return KillResult("killed", "bridge killed", [], True)
    finally:
        masked: dict[int, object] = {}
        if on_main:
            for sig in stop_signals:
                try:
                    masked[sig] = signal.signal(sig, signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
        try:
            _hook("finally_start")
            _release_tree(suspended, remembered, root_kill_started)
            if not root_kill_started:
                _cas_intent(persona_dir, nonce, lambda _cur: None)
                # A kill called off before it killed withdraws its hard-exit request too (review of #458): the bridge
                # still runs and would act on it later, after the user cancelled.
                if request_written and process_identity.names_this_bridge(
                    shutdown_progress.read_record(persona_dir, shutdown_progress.REQUEST_FILENAME), st.pid, st.pid_token
                ):
                    shutdown_progress.delete_record(persona_dir, shutdown_progress.REQUEST_FILENAME)
        finally:
            if on_main:
                for sig, handler in prev.items():
                    try:
                        signal.signal(sig, handler)  # type: ignore[arg-type]
                    except (ValueError, OSError):
                        pass


def kill_command(pid: int, token: str | None, *, timeout: float = DEFAULT_TIMEOUT_S) -> KillResult:
    """End one running command (a CLI process that holds or may take the engine lock) and its descendants (HS3 of #633).

    The record that names it (``engine_files.live_command_runs``) gives ``pid`` and ``token``. The ``psutil.Process`` is created
    first (``psutil.Error``: not running); nothing is killed unless the process still carries ``token``. A record with no token
    is not asked: ``process_identity.matches(pid, None)`` answers GONE for a live process, which a caller would count as success,
    so a falsy token, like an UNKNOWN answer, is ``unknown_identity`` (as ``daemon.cmd_stop`` treats an unconfirmed identity). The
    descendants are remembered first, then ``_end_tree`` freezes and kills the tree with ``timeout`` as its poll bound, and the
    ``finally`` calls ``_release_tree``. No intent, hard-exit request or service manager is involved: a command has no graceful
    stop to wait for (S107: "whatever is running gets shut down"; S110's no-automatic-hard-kill is about a bridge told to stop).

    Accepted edge (P4): no signal handler is installed here. Ctrl-C raises ``KeyboardInterrupt`` and the ``finally`` resumes the
    tree (before the root's kill was started) or finishes it (from then on, an interrupt inside the root's kill call included: the
    root is killed with its descendants). SIGTERM or SIGHUP end the starting process at once and leave a tree that was already frozen stopped
    until the next start's ``kill_command`` ends it; nothing is lost, the frozen processes only wait. On Windows the end code is 0
    (``_terminate_exit_0``).

    Accepted edge (P4, B57 HSC (5)): a kill of the MCP child (a descendant of a bridge's claude call, so of a command's tree)
    between its memory commit and its feelings hook leaves that memory saved without its feelings applied. The kill happens before
    the rebuild child starts, so a rebuild that then succeeds folds the row; one that fails falls under the existing F-DA edge
    (engine.py:393-396, S39).
    """
    import psutil

    from brain import process_identity

    try:
        proc = psutil.Process(pid)
    except psutil.Error:
        return KillResult("not_running", "command not running")
    if not token:
        return KillResult("unknown_identity", f"identity of command {pid} could not be confirmed; not killing")
    live = process_identity.matches(pid, token)
    if live == process_identity.UNKNOWN:
        return KillResult("unknown_identity", f"identity of command {pid} could not be confirmed; not killing")
    if live != "alive":
        return KillResult("not_running", "command not running")

    remembered: dict[int, object] = {p.pid: p for p in _children(proc)}
    suspended: list = []
    root_kill_started = False

    def _root_killed() -> None:
        nonlocal root_kill_started
        root_kill_started = True

    try:
        survivors = _end_tree(proc, suspended, remembered, timeout, on_root_killed=_root_killed)
    finally:
        _release_tree(suspended, remembered, root_kill_started)
    if survivors:
        return KillResult("killed", f"processes still running after the kill: {survivors}", survivors, True)
    return KillResult("killed", "command killed", [], True)

