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
from collections.abc import Iterator
from pathlib import Path

from brain.bridge.state_file import pid_is_alive
from brain.paths import get_home

STATE_FILE = "current.json"
LOCK_FILE = ".lock"


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
    os.replace(tmp, root / STATE_FILE)


def activate(root: Path, entry: dict) -> None:
    old = read_state(root)["active"]
    previous = old if old and old.get("dir") != entry["dir"] else read_state(root)["previous"]
    _write_state(root, entry, previous)


def revert(root: Path) -> None:
    _write_state(root, None, read_state(root)["active"])


def rollback(root: Path) -> None:
    _write_state(root, read_state(root)["previous"], None)


def prune(root: Path) -> None:
    state = read_state(root)
    keep = {e["dir"] for e in (state["active"], state["previous"]) if e and e.get("dir")}
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)


@contextlib.contextmanager
def overlay_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_FILE
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                holder = int(path.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError):
                holder = 0
            if holder and pid_is_alive(holder):
                raise OverlayBusy(f"an update is already running (pid {holder})") from None
            path.unlink(missing_ok=True)  # stale: its process is gone
    else:
        raise OverlayBusy("could not take the overlay lock")
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)
