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

The ``finally``: while the bridge's kill call has not returned, resume everything suspended and delete the intent
(only if it still holds this nonce); after it has returned, kill the frozen and remembered descendants and keep the
committed intent (the respawn guard honours it). Stop signals are ignored while the ``finally`` runs.
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


def _service_backend(platform: str | None = None):
    platform = platform or sys.platform
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
    platform = platform or sys.platform
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
    platform = platform or sys.platform
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
    platform = platform or sys.platform
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
        if sys.platform.startswith("win"):  # pragma: no cover - Windows CI leg
            _terminate_exit_0(proc.pid)
        else:
            proc.kill()
    except Exception:  # noqa: BLE001 - gone already
        pass


def _terminate_exit_0(pid: int) -> None:  # pragma: no cover - Windows CI leg
    """Windows: end the process with exit code 0, so Task Scheduler does not treat the end as a failure."""
    import ctypes

    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = k32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if handle:
        try:
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
    """Has another bridge started for this persona meanwhile? (the record names another ALIVE pid)."""
    st = state_file.read(persona_dir)
    if st is None or st.pid is None or st.pid == target_pid:
        return False
    return state_file.bridge_liveness(st, persona_dir) == state_file.ALIVE


# ---------------------------------------------------------------- the helper


def kill_bridge(
    persona_dir: Path,
    persona: str | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    skip_hard_exit_request: bool = False,
    _hooks: dict | None = None,
) -> KillResult:
    """Kill the persona's bridge and its tree (module docstring). ``_hooks`` are test seams (named step callbacks)."""
    import psutil

    from brain import process_identity

    hooks = _hooks or {}
    persona_dir = Path(persona_dir)
    persona = persona or persona_dir.name
    st = state_file.read(persona_dir)
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

    stop_signals = [signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
    nonce = secrets.token_hex(8)

    remembered: dict[int, object] = {}
    suspended: list = []
    kill_returned = False

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

        # Steps 4-5: service ownership and the graceful service path.
        if service_installed(persona) and owned_by_service(persona, st.pid, st.client_origin):
            if not legacy and not skip_hard_exit_request:
                shutdown_progress.write_record(
                    persona_dir,
                    shutdown_progress.REQUEST_FILENAME,
                    shutdown_progress.hard_exit_request(
                        pid=st.pid, pid_token=st.pid_token, requested_at=datetime.now(UTC).isoformat()
                    ),
                )
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

        frozen = False
        if _alive(bridge):
            try:
                bridge.suspend()
                frozen = True
            except psutil.NoSuchProcess:
                pass  # it ended between the check and the suspend (review S4): the "already gone" branch below
        if frozen:
            suspended.append(bridge)
            _hook("after_bridge_suspend")
            # Every pid tried is remembered, suspended or not (review M3): a descendant whose suspend() fails and
            # that stays listed cannot keep this loop going; it is killed with the tree below.
            tried = {bridge.pid}
            while True:
                new = [p for p in _children(bridge) if p.pid not in tried]
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
            _kill_proc(bridge)
            kill_returned = True
            _hook("after_bridge_kill")
            for p in suspended[1:]:
                _kill_proc(p)
        else:
            kill_returned = True
        for p in remembered.values():
            if _alive(p):
                _kill_proc(p)

        # Step 7: poll gone over everything.
        everyone = [bridge, *remembered.values()]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(_alive(p) for p in everyone):
            time.sleep(POLL_S)
        survivors = [p.pid for p in everyone if _alive(p)]
        for p in everyone:
            if _alive(p):
                _kill_proc(p)
        if survivors:
            return KillResult("killed", f"processes still running after the kill: {survivors}", survivors, True)
        return KillResult("killed", "bridge killed", [], True)
    finally:
        masked: dict[int, object] = {}
        if on_main:
            for sig in [*stop_signals, signal.SIGINT]:
                try:
                    masked[sig] = signal.signal(sig, signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
        try:
            _hook("finally_start")
            if not kill_returned:
                for p in suspended:
                    try:
                        p.resume()
                    except Exception:  # noqa: BLE001
                        pass
                _cas_intent(persona_dir, nonce, lambda _cur: None)
            else:
                for p in [*suspended[1:], *remembered.values()]:
                    if _alive(p):
                        _kill_proc(p)
        finally:
            if on_main:
                for sig, handler in prev.items():
                    try:
                        signal.signal(sig, handler)  # type: ignore[arg-type]
                    except (ValueError, OSError):
                        pass
                if signal.SIGINT in masked:
                    try:
                        signal.signal(signal.SIGINT, masked[signal.SIGINT])  # type: ignore[arg-type]
                    except (ValueError, OSError):
                        pass

