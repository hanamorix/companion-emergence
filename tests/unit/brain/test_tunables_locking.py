"""The tunables lock (#529: C8 a/c, C42 e): no lost update across the file's read-modify-write accessors.

Real locks everywhere; the competing write is injected deterministically at the point just before the accessor's own
write (the module-level window hook), from a thread that must WAIT for the lock to get in. The legacy-JSON fold's
interleavings (C44) live in ``test_tunables_migration.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.tunables_file import FILE_NAME, read_values, write_text


@pytest.fixture()
def t(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as mod
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 5.0)
    mod._reset_for_tests()
    yield mod
    mod._reset_for_tests()


def _file(tmp_path: Path) -> Path:
    return tmp_path / FILE_NAME


class _Competitor:
    """A thread that performs ``set_values`` when started and reports whether it finished within a short join."""

    def __init__(self, t, updates: dict) -> None:
        self._thread = threading.Thread(target=lambda: t.set_values(updates))

    def start_and_wait_briefly(self) -> bool:
        self._thread.start()
        self._thread.join(timeout=1.0)
        return not self._thread.is_alive()

    def finish(self) -> None:
        self._thread.join(timeout=20)
        assert not self._thread.is_alive()


def test_set_values_while_sync_file_holds_its_window_both_survive(t, tmp_path) -> None:
    """C8 (a): the competitor blocks on the lock, then lands after the sync write; neither update is lost."""
    t.register("demo.key", 1, description="A demo.")
    write_text(tmp_path, '"seed" = 1\n')
    competitor = _Competitor(t, {"competitor": 2})
    finished_inside: list[bool] = []

    def hook() -> None:
        finished_inside.append(competitor.start_and_wait_briefly())

    t._LOCKED_WINDOW_HOOK = hook
    try:
        t.sync_file()
    finally:
        t._LOCKED_WINDOW_HOOK = None
    competitor.finish()
    assert finished_inside == [False]  # it could not get in while the window was open
    assert read_values(tmp_path) == {"seed": 1, "competitor": 2, "demo.key": 1}


def test_the_lock_hook_is_called_inside_the_window_just_before_the_write_by_set_values(t, tmp_path) -> None:
    write_text(tmp_path, '"a" = 1\n')
    seen: list[bytes] = []
    t._LOCKED_WINDOW_HOOK = lambda: seen.append(_file(tmp_path).read_bytes())
    try:
        t.set_values({"a": 2})
    finally:
        t._LOCKED_WINDOW_HOOK = None
    assert seen == [b'"a" = 1\n']  # called before the replace, so the old file is still on disk


_WORKER = """
import sys
sys.path.insert(0, {root!r})
from brain import tunables, dev_constants
dev_constants.SETTINGS_FILE_LOCK_WAIT_S = 60.0
tag = sys.argv[1]
for i in range(50):
    tunables.set_values({{f"{{tag}}{{i}}": i}})
"""


def test_two_processes_writing_different_keys_lose_nothing(tmp_path, repo_root) -> None:
    """C8 (c): 50 rounds in each of two processes against one file: every one of the 100 keys is present."""
    code = _WORKER.format(root=str(repo_root))
    env = {**os.environ, "KINDLED_HOME": str(tmp_path)}
    procs = [subprocess.Popen([sys.executable, "-c", code, tag], env=env) for tag in ("a", "b")]
    for p in procs:
        assert p.wait(timeout=120) == 0
    values = read_values(tmp_path)
    # the creating write also carries the registered keys (the lull key), so check inclusion, not equality
    assert {f"a{i}" for i in range(50)} | {f"b{i}" for i in range(50)} <= set(values)
    assert {k for k in values if k not in {f"a{i}" for i in range(50)} | {f"b{i}" for i in range(50)}} == {"chat.idle_lull_seconds"}


# ---------------------------------------------------------------------------
# C42 (e): callers on a wedged lock do not write
# ---------------------------------------------------------------------------


def test_set_values_on_a_wedged_lock_raises_and_leaves_the_file(t, tmp_path, wedged_lock, monkeypatch) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    write_text(tmp_path, '"a" = 1\n')
    before = _file(tmp_path).read_bytes()
    with wedged_lock(_file(tmp_path)), pytest.raises(t.TunablesWriteError):
        t.set_values({"a": 2})
    assert _file(tmp_path).read_bytes() == before


def test_sync_file_on_a_wedged_lock_returns_without_writing(t, tmp_path, wedged_lock, monkeypatch, caplog) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    t.register("new.key", 1, description="A new thing.")
    write_text(tmp_path, '"a" = 1\n')
    before = _file(tmp_path).read_bytes()
    stamp = _file(tmp_path).stat().st_mtime_ns
    with wedged_lock(_file(tmp_path)), caplog.at_level("WARNING"):
        t.sync_file()
    assert _file(tmp_path).read_bytes() == before
    assert _file(tmp_path).stat().st_mtime_ns == stamp
    assert sum("lock" in r.getMessage().lower() for r in caplog.records) >= 1


def test_sync_file_writes_nothing_else_while_holding_the_lock(t, tmp_path, monkeypatch) -> None:
    """The longest hold is one read, one atomic write and its fsync (K = 0 other replaces): record every write in the window."""
    from brain.utils import toml_io

    t.register("a", 1, description="A.")
    writes: list[str] = []
    real = toml_io.write_toml_text
    monkeypatch.setattr(toml_io, "write_toml_text", lambda p, d, **kw: (writes.append(Path(p).name), real(p, d, **kw))[1])
    monkeypatch.setattr(t, "_write_marker", lambda record: writes.append("notice"))
    t.sync_file()
    assert writes == [FILE_NAME]
