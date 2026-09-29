"""The brain overlay's on-disk state (#286 §3.3).

`<home>/brain-overlay/current.json` holds `{"active": entry|None, "previous":
entry|None}`; an entry is `{"dir", "commit", "brain_version", "bundle_id"}`. The
activation hook (overlay_hook.py) reads it at every interpreter start, so every
write is atomic (temp file + os.replace). Read with .get(): the file must stay
readable by older and newer brains (#286 §7).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

from brain.bridge.state_file import pid_is_alive
from brain.paths import get_home

STATE_FILE = "current.json"
LOCK_FILE = ".lock"
# <overlay folder>/.in-use/<pid>: a process running `brain` from that folder (#302).
IN_USE_DIR = ".in-use"
_RETRY_ATTEMPTS = 6
_RETRY_DELAY_S = 0.05  # doubling: ~1.5 s in all before the error is let through


class OverlayBusy(RuntimeError):  # noqa: N818
    """Another live process is updating the overlay."""


def overlay_root() -> Path:
    return get_home() / "brain-overlay"


def read_state(root: Path) -> dict:
    try:
        data = json.loads((root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"active": None, "previous": None}
    if not isinstance(data, dict):
        return {"active": None, "previous": None}
    return {"active": data.get("active") or None, "previous": data.get("previous") or None}


def _write_state(root: Path, active: dict | None, previous: dict | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{STATE_FILE}.tmp"
    tmp.write_text(json.dumps({"active": active, "previous": previous}, indent=2), encoding="utf-8")
    replace_retrying(tmp, root / STATE_FILE)


def replace_retrying(src: Path, dst: Path) -> None:
    """os.replace, riding out a transient PermissionError — on Windows, AV scanning
    freshly written files or another interpreter's hook reading current.json at
    start-up (#302). Anything else, or a lock that outlasts the retries, raises."""
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_DELAY_S * 2**attempt)


def mark_in_use(root: Path | None = None) -> None:
    """Record that THIS process runs `brain` from an overlay folder, so prune leaves
    the folder alone while the process lives (#302). No-op on the release brain.
    Called once at bridge start; prune ignores a dead process's marker."""
    import brain

    root = (root or overlay_root()).resolve()
    try:
        rel = Path(brain.__file__).resolve().relative_to(root)
    except ValueError:
        return
    if len(rel.parts) < 2:
        return
    marks = root / rel.parts[0] / IN_USE_DIR
    marks.mkdir(exist_ok=True)
    (marks / str(os.getpid())).write_text("", encoding="utf-8")


def in_use(folder: Path) -> bool:
    """A live process runs `brain` from this folder (it wrote an .in-use marker)."""
    # ponytail: a reused pid keeps a folder one prune longer — the safe direction.
    try:
        return any(m.name.isdigit() and pid_is_alive(int(m.name)) for m in (folder / IN_USE_DIR).iterdir())
    except OSError:
        return False


def activate(root: Path, entry: dict) -> None:
    old = read_state(root)["active"]
    previous = old if old and old.get("dir") != entry["dir"] else read_state(root)["previous"]
    _write_state(root, entry, previous)


def revert(root: Path) -> None:
    state = read_state(root)
    if state["active"] is None:
        return  # already reverted: don't clobber `previous` with the None it left behind
    _write_state(root, None, state["active"])


def rollback(root: Path) -> None:
    state = read_state(root)
    if state["previous"] is None:
        # nothing older to go back to: the release brain, keeping this overlay as the
        # rollback target so prune doesn't delete it
        revert(root)
        return
    _write_state(root, state["previous"], None)


def prune(root: Path) -> None:
    state = read_state(root)
    keep = {e["dir"] for e in (state["active"], state["previous"]) if e and e.get("dir")}
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name not in keep and not in_use(child):
            shutil.rmtree(child, ignore_errors=True)


@contextlib.contextmanager
def overlay_lock(root: Path) -> Iterator[None]:
    """Hold `<root>/.lock` for the duration. The lock file is published atomically
    (pid written to a private temp file, then os.link into place), so it is never
    observable empty; release removes it only if it still names this process."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_FILE
    me = str(os.getpid())
    tmp = root / f".lock.{me}.tmp"
    tmp.write_text(me, encoding="utf-8")
    try:
        for _ in range(2):
            try:
                os.link(tmp, path)  # atomic; FileExistsError while someone holds it
                break
            except FileExistsError:
                try:
                    holder = int(path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    holder = 0
                if holder and pid_is_alive(holder):
                    raise OverlayBusy(f"an update is already running (pid {holder}; lock {path})") from None
                # ponytail: best-effort stale takeover, like bridge.daemon.acquire_lock —
                # two takers racing on the same dead lock at the same instant can still collide.
                path.unlink(missing_ok=True)
        else:
            raise OverlayBusy("could not take the overlay lock")
    finally:
        tmp.unlink(missing_ok=True)
    try:
        yield
    finally:
        try:
            if path.read_text(encoding="utf-8").strip() == me:
                path.unlink()
        except OSError:
            pass
