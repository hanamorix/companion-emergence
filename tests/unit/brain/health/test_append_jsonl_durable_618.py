"""#618: an unclean machine stop must not leave a just-appended JSONL record as NUL bytes."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from brain.health.json_io import append_jsonl, iter_jsonl


def _crash_after(monkeypatch, path: Path) -> Callable[[], None]:
    """Record the file size at every fsync; ``crash()`` NULs every byte past the largest synced size."""
    real_fsync = os.fsync
    synced = [0]

    def spy(fd: int) -> None:
        synced[0] = max(synced[0], os.fstat(fd).st_size)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)

    def crash() -> None:
        data = path.read_bytes()
        keep = data[: synced[0]]
        path.write_bytes(keep + b"\0" * (len(data) - len(keep)))

    return crash


def test_a_default_append_survives_an_unclean_stop(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "log.jsonl"
    crash = _crash_after(monkeypatch, p)
    append_jsonl(p, {"n": 1})
    crash()
    assert list(iter_jsonl(p)) == [{"n": 1}]
