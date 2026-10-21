"""The terminal stop watcher (#456 Q-A, D18): what `nell supervisor stop` / `restart` show while the bridge stops.

It reads only files and the bridge's identity (the bridge's HTTP listener is closed once a stop begins): the progress
record of THIS bridge (pid + token; a record of another process is ignored), and whether the remembered pid + token is
gone. It prints the remaining tasks when they change; at the recorded stop request + 180 s it offers a hard exit (with
a terminal only); a "y" writes the hard-exit request, and a request the bridge does not act on within ``--timeout``
escalates to the kill helper. Without a terminal it never prompts and never kills: it returns 1 after ``--timeout``.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge import shutdown_progress

POLL_S = 0.2
# Owner Q1 (2026-10-19): the hard exit is offered 180 s after the stop was requested. The app uses the same value.
HARD_EXIT_OFFER_AFTER_S = 180.0


class _Clock:
    def now(self) -> float:
        return time.time()

    def sleep(self, dt: float) -> None:
        time.sleep(dt)


_clock = _Clock()


def _liveness(pid: int, token: str | None) -> str:
    """"alive" / "gone" / "unknown" for the remembered bridge (UNKNOWN keeps waiting)."""
    from brain import process_identity

    if token:
        return process_identity.matches(pid, token)
    from brain.bridge import state_file

    return "alive" if state_file.pid_is_alive(pid) else "gone"


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _prompt(text: str) -> str:
    return input(text)


def _plural(n: int) -> str:
    return "task" if n == 1 else "tasks"


def progress_line(tasks: list[str]) -> str:
    n = len(tasks)
    return f"Stopping. Waiting for {n} {_plural(n)} to finish: {', '.join(tasks)}."


def offer_text(tasks: list[str]) -> str:
    n = len(tasks)
    minutes = int(HARD_EXIT_OFFER_AFTER_S // 60)
    return (
        f"Still waiting for {n} {_plural(n)} after {minutes} minutes: {', '.join(tasks)}. Hard exit now? "
        "The next start will pick up anything left unfinished. [y/N] "
    )


EVERYTHING_STOPPED = "Everything has stopped."
HARD_EXIT_DONE = "Hard exit done. The next start will pick up anything left unfinished."


def _requested_at(rec: dict | None, persona_dir: Path, fallback: float) -> float:
    for value in ((rec or {}).get("requested_at"), _record_requested_at(persona_dir)):
        if value:
            try:
                return datetime.fromisoformat(str(value)).timestamp()
            except ValueError:
                continue
    return fallback


def _record_requested_at(persona_dir: Path) -> str | None:
    from brain.bridge import state_file

    try:
        st = state_file.read(persona_dir)
    except Exception:  # noqa: BLE001
        return None
    return st.stop_requested_at if st is not None else None


def watch_stop(
    persona_dir: Path,
    *,
    pid: int,
    token: str | None,
    timeout: float,
    interactive: bool,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    prompt: Callable[[str], str] | None = None,
    out: Callable[[str], object] = print,
    kill: Callable[..., object] | None = None,
) -> int:
    """Watch the bridge (pid + token) until it is gone. Returns 0 when it is gone (stopped, hard-exited or killed),
    1 without a terminal after ``timeout`` (nothing killed)."""
    now = now or _clock.now
    sleep = sleep or _clock.sleep
    prompt = prompt or _prompt
    if kill is None:
        from brain.bridge import kill as kill_mod

        kill = kill_mod.kill_bridge
    persona_dir = Path(persona_dir)
    started = now()
    last_tasks: list[str] | None = None
    offered = False
    answer: dict[str, str] = {}
    request_sent_at: float | None = None
    while True:
        live = _liveness(pid, token)
        if live == "gone":
            out(HARD_EXIT_DONE if request_sent_at is not None else EVERYTHING_STOPPED)
            return 0
        rec = shutdown_progress.read_record(persona_dir, shutdown_progress.PROGRESS_FILENAME)
        mine = rec if shutdown_progress.names_this_bridge(rec, pid, token) else None
        tasks = list(mine.get("tasks") or []) if mine else None
        if tasks is not None and tasks != last_tasks:
            out(progress_line(tasks) if tasks else "Stopping. Finishing up.")
            last_tasks = tasks
        t = now()
        if (
            interactive
            and not offered
            and t >= _requested_at(mine, persona_dir, started) + HARD_EXIT_OFFER_AFTER_S
        ):
            offered = True
            text = offer_text(last_tasks or [])

            def _ask(text: str = text) -> None:
                try:
                    answer["value"] = prompt(text)
                except (EOFError, OSError):
                    answer["value"] = ""

            threading.Thread(target=_ask, daemon=True).start()
        if request_sent_at is None and answer.get("value", "").strip().lower() in ("y", "yes"):
            shutdown_progress.write_record(
                persona_dir,
                shutdown_progress.REQUEST_FILENAME,
                shutdown_progress.hard_exit_request(
                    pid=pid, pid_token=token, requested_at=datetime.now(UTC).isoformat()
                ),
            )
            request_sent_at = t
        if request_sent_at is not None and t >= request_sent_at + timeout:
            out("The bridge did not act on the hard exit; killing it.")
            result = kill(persona_dir=persona_dir, timeout=timeout, skip_hard_exit_request=True)
            out(result.message)
            return 0 if result.succeeded else 1
        if not interactive and t >= started + timeout:
            out(f"The bridge is still stopping after {int(timeout)} s.")
            return 1
        sleep(POLL_S)
