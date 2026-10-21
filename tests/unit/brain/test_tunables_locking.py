"""The tunables lock (#529: C8 a-c, C42 e, C44): no lost update across the file's read-modify-write accessors.

Real locks everywhere; the competing write is injected deterministically at the point just before the accessor's own
write (the module-level window hook), from a thread that must WAIT for the lock to get in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest


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
    return tmp_path / "tunables.json"


def _read(tmp_path: Path) -> dict:
    return json.loads(_file(tmp_path).read_text(encoding="utf-8"))


class _Competitor:
    """A thread that performs ``set_overrides`` when started and reports whether it finished within a short join."""

    def __init__(self, t, updates: dict) -> None:
        self._t, self._updates = t, updates
        self._thread = threading.Thread(target=lambda: t.set_overrides(updates))

    def start_and_wait_briefly(self) -> bool:
        self._thread.start()
        self._thread.join(timeout=1.0)
        return not self._thread.is_alive()

    def finish(self) -> None:
        self._thread.join(timeout=20)
        assert not self._thread.is_alive()


def test_set_overrides_while_write_defaults_section_holds_its_window_both_survive(t, tmp_path) -> None:
    """C8 (a): the competitor blocks on the lock, then lands after the defaults write; neither update is lost."""
    t.register("demo.key", 1)
    t.set_overrides({"seed": 1})
    competitor = _Competitor(t, {"competitor": 2})
    finished_inside = []

    def hook() -> None:
        finished_inside.append(competitor.start_and_wait_briefly())

    t._LOCKED_WINDOW_HOOK = hook
    try:
        t.write_defaults_section()
    finally:
        t._LOCKED_WINDOW_HOOK = None
    competitor.finish()
    assert finished_inside == [False]  # it could not get in while the window was open
    data = _read(tmp_path)
    assert data["overrides"] == {"seed": 1, "competitor": 2}
    assert data["defaults"].get("demo.key") == 1


def test_set_overrides_during_migrate_idle_keys_both_survive(t, tmp_path) -> None:
    """C8 (b) and C44: an external set_overrides that lands inside the migration's read-to-write window survives."""
    from brain import tunables_migration as tm

    _file(tmp_path).write_text(
        json.dumps({"overrides": {"chat.pass2_min_idle_seconds": 10, "keep": 1}}), encoding="utf-8"
    )
    competitor = _Competitor(t, {"model.interactive_chat": "claude-opus-5"})
    finished_inside = []

    def hook() -> None:
        finished_inside.append(competitor.start_and_wait_briefly())

    tm._LOCKED_WINDOW_HOOK = hook
    try:
        tm.migrate_idle_keys(tmp_path)
    finally:
        tm._LOCKED_WINDOW_HOOK = None
    competitor.finish()
    assert finished_inside == [False]
    assert _read(tmp_path)["overrides"] == {"keep": 1, "model.interactive_chat": "claude-opus-5"}


def test_a_failed_verify_restores_without_clobbering_a_concurrent_write(t, tmp_path, monkeypatch) -> None:
    """C44: the lock covers read through verify AND restore, so a concurrent write waits for the restore."""
    from brain import tunables_migration as tm

    original = json.dumps({"overrides": {"chat.pass2_min_idle_seconds": 10, "keep": 1}}).encode()
    _file(tmp_path).write_bytes(original)
    competitor = _Competitor(t, {"late": 3})

    def failing_reload(*args, **kwargs):
        raise ValueError("forced verify failure")

    def hook() -> None:
        competitor.start_and_wait_briefly()

    monkeypatch.setattr(tm, "read_json", failing_reload)
    tm._LOCKED_WINDOW_HOOK = hook
    try:
        tm.migrate_idle_keys(tmp_path)
    finally:
        tm._LOCKED_WINDOW_HOOK = None
    competitor.finish()
    overrides = _read(tmp_path)["overrides"]
    assert overrides["late"] == 3 and overrides["keep"] == 1
    assert "chat.pass2_min_idle_seconds" in overrides  # restored: the failed migration left the file as it was


def test_an_ordinary_boot_takes_no_lock_and_leaves_the_file_untouched(t, tmp_path, wedged_lock, monkeypatch) -> None:
    """C44: no retired key means the cheap pre-check returns before any lock (a wedged holder does not matter)."""
    from brain import dev_constants
    from brain import tunables_migration as tm

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.2)
    _file(tmp_path).write_text(json.dumps({"overrides": {"keep": 1}}), encoding="utf-8")
    before = _file(tmp_path).stat().st_mtime_ns
    with wedged_lock(_file(tmp_path)):
        tm.migrate_idle_keys(tmp_path)
    assert _file(tmp_path).stat().st_mtime_ns == before


def test_the_bytes_written_back_come_from_the_reread_inside_the_lock(t, tmp_path) -> None:
    """C44: a key added between the pre-check read and the lock is in the written file (never the pre-check read)."""
    from brain import tunables_migration as tm

    _file(tmp_path).write_text(json.dumps({"overrides": {"chat.pass2_min_idle_seconds": 10}}), encoding="utf-8")
    real_read_bytes = Path.read_bytes
    calls = {"n": 0}

    def racing_read_bytes(self):
        data = real_read_bytes(self)
        calls["n"] += 1
        if calls["n"] == 1 and self.name == "tunables.json":
            # a writer lands right after the pre-check read, before the migration takes its lock
            t.set_overrides({"landed_late": 9})
        return data

    Path.read_bytes = racing_read_bytes  # type: ignore[method-assign]
    try:
        tm.migrate_idle_keys(tmp_path)
    finally:
        Path.read_bytes = real_read_bytes  # type: ignore[method-assign]
    assert _read(tmp_path)["overrides"] == {"landed_late": 9}


_WORKER = """
import sys
sys.path.insert(0, {root!r})
from brain import tunables, dev_constants
dev_constants.SETTINGS_FILE_LOCK_WAIT_S = 60.0
tag = sys.argv[1]
for i in range(50):
    tunables.set_overrides({{f"{{tag}}{{i}}": i}})
"""


def test_two_processes_writing_different_keys_lose_nothing(tmp_path, repo_root) -> None:
    """C8 (c): 50 rounds in each of two processes against one file: every one of the 100 keys is present."""
    code = _WORKER.format(root=str(repo_root))
    env = {**os.environ, "KINDLED_HOME": str(tmp_path)}
    procs = [subprocess.Popen([sys.executable, "-c", code, tag], env=env) for tag in ("a", "b")]
    for p in procs:
        assert p.wait(timeout=120) == 0
    overrides = _read(tmp_path)["overrides"]
    assert {f"a{i}" for i in range(50)} | {f"b{i}" for i in range(50)} == set(overrides)


# ---------------------------------------------------------------------------
# C42 (e): callers on a wedged lock do not write
# ---------------------------------------------------------------------------


def test_set_overrides_on_a_wedged_lock_raises_and_leaves_the_file(t, tmp_path, wedged_lock, monkeypatch) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    _file(tmp_path).write_text(json.dumps({"overrides": {"a": 1}}), encoding="utf-8")
    before = _file(tmp_path).read_bytes()
    with wedged_lock(_file(tmp_path)), pytest.raises(t.TunablesWriteError):
        t.set_overrides({"a": 2})
    assert _file(tmp_path).read_bytes() == before


def test_write_defaults_and_idle_migration_on_a_wedged_lock_return_without_writing(
    t, tmp_path, wedged_lock, monkeypatch, caplog
) -> None:
    from brain import dev_constants
    from brain import tunables_migration as tm

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    _file(tmp_path).write_text(
        json.dumps({"overrides": {"chat.pass2_min_idle_seconds": 10}}), encoding="utf-8"
    )
    before = _file(tmp_path).read_bytes()
    stamp = _file(tmp_path).stat().st_mtime_ns
    with wedged_lock(_file(tmp_path)):
        with caplog.at_level("WARNING"):
            t.write_defaults_section()
            tm.migrate_idle_keys(tmp_path)
    assert _file(tmp_path).read_bytes() == before
    assert _file(tmp_path).stat().st_mtime_ns == stamp
    assert sum("lock" in r.getMessage().lower() for r in caplog.records) >= 2
