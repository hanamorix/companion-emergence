"""``tunables.set_overrides`` and the hardened reader (#529: C7, C9, C36)."""

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

    mod._reset_for_tests()
    yield mod
    mod._reset_for_tests()


def _file(tmp_path: Path) -> Path:
    return tmp_path / "tunables.json"


def _read(tmp_path: Path) -> dict:
    return json.loads(_file(tmp_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# C7: what the writer changes and what it leaves alone
# ---------------------------------------------------------------------------


def test_only_the_named_keys_change_and_the_rest_is_kept_as_parsed(t, tmp_path) -> None:
    _file(tmp_path).write_text(
        json.dumps({"_readme": "r", "defaults": {"a": 1}, "overrides": {"a": 5, "b": 2}, "extra": [1]}),
        encoding="utf-8",
    )
    assert t.set_overrides({"a": 9}) == {"a": 9}
    data = _read(tmp_path)
    assert data["overrides"] == {"a": 9, "b": 2}
    assert data["_readme"] == "r" and data["defaults"] == {"a": 1} and data["extra"] == [1]


def test_none_deletes_a_key_and_reports_it_written(t, tmp_path) -> None:
    _file(tmp_path).write_text(json.dumps({"overrides": {"a": 1, "b": 2}}), encoding="utf-8")
    assert t.set_overrides({"a": None}) == {"a": None}
    assert _read(tmp_path)["overrides"] == {"b": 2}


def test_deleting_an_absent_key_writes_nothing(t, tmp_path) -> None:
    """No file churn: neither a missing file is created nor an existing file touched (no lock round trip either)."""
    assert t.set_overrides({"a": None}) == {}
    assert not _file(tmp_path).exists()
    _file(tmp_path).write_text(json.dumps({"overrides": {"b": 2}}), encoding="utf-8")
    before = _file(tmp_path).stat().st_mtime_ns
    assert t.set_overrides({"a": None}) == {}
    assert _file(tmp_path).stat().st_mtime_ns == before


def test_an_invalid_text_in_a_model_key_survives_an_unrelated_write(t, tmp_path) -> None:
    _file(tmp_path).write_text(json.dumps({"overrides": {"model.interactive_chat": "banana"}}), encoding="utf-8")
    t.set_overrides({"other.key": 3})
    assert _read(tmp_path)["overrides"] == {"model.interactive_chat": "banana", "other.key": 3}


def test_only_missing_skips_existing_keys_and_returns_only_what_it_wrote(t, tmp_path) -> None:
    _file(tmp_path).write_text(json.dumps({"overrides": {"a": 1}}), encoding="utf-8")
    written = t.set_overrides({"a": 2, "b": 3}, only_missing=True)
    assert written == {"b": 3}
    assert _read(tmp_path)["overrides"] == {"a": 1, "b": 3}


def test_a_zero_byte_file_is_no_overrides_for_the_reader_and_the_writer(t, tmp_path) -> None:
    _file(tmp_path).write_bytes(b"")
    assert t.get_tunable("a", 7) == 7
    assert t.set_overrides({"a": 8}) == {"a": 8}
    assert _read(tmp_path)["overrides"] == {"a": 8}


@pytest.mark.parametrize(
    "content",
    [b"{not json", b'{"overrides": [1, 2]}', b"[1, 2]", b"\xff\xfe\x00"],
)
def test_a_damaged_file_raises_and_is_not_modified(t, tmp_path, content) -> None:
    _file(tmp_path).write_bytes(content)
    with pytest.raises(t.TunablesWriteError):
        t.set_overrides({"a": 1})
    assert _file(tmp_path).read_bytes() == content
    assert not list(tmp_path.glob("tunables.json.corrupt-*"))  # a writer never heals


def test_an_unreadable_file_raises_and_is_not_modified(t, tmp_path, monkeypatch) -> None:
    _file(tmp_path).write_text(json.dumps({"overrides": {"a": 1}}), encoding="utf-8")
    before = _file(tmp_path).read_bytes()

    def boom(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("brain.health.json_io._read_bytes", boom)
    with pytest.raises(t.TunablesWriteError):
        t.set_overrides({"a": 2})
    monkeypatch.undo()
    assert _file(tmp_path).read_bytes() == before


def test_an_absent_file_is_created_with_the_standard_skeleton(t, tmp_path) -> None:
    assert t.set_overrides({"a": 1}) == {"a": 1}
    data = _read(tmp_path)
    assert data["overrides"] == {"a": 1}
    assert isinstance(data["_readme"], str) and data["defaults"] == {}


def test_a_reader_polling_during_many_writes_never_sees_a_torn_or_default_value(t, tmp_path) -> None:
    """Replacement is atomic: an unchanged key never flickers to its default while another key is written 200 times."""
    t.set_overrides({"keep": 7})
    stop = threading.Event()
    seen: list[object] = []

    def reader() -> None:
        while not stop.is_set():
            seen.append(t.get_tunable("keep", 0))

    th = threading.Thread(target=reader)
    th.start()
    try:
        for i in range(200):
            t.set_overrides({"moving": i})
    finally:
        stop.set()
        th.join()
    assert seen and set(seen) == {7}


# ---------------------------------------------------------------------------
# C9: live, lock-free, and not stale across processes
# ---------------------------------------------------------------------------


def test_a_write_is_visible_in_process_at_once(t) -> None:
    assert t.get_tunable("a", 1) == 1
    t.set_overrides({"a": 2})
    assert t.get_tunable("a", 1) == 2


def test_two_contents_with_identical_mtime_ns_are_both_seen(t, tmp_path) -> None:
    """The cache key is (mtime_ns, size, inode): a coarse timestamp tick cannot hide a second write."""
    path = _file(tmp_path)
    path.write_text(json.dumps({"overrides": {"a": 1}}), encoding="utf-8")
    fixed = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed, fixed))
    assert t.get_tunable("a", 0) == 1
    path.write_text(json.dumps({"overrides": {"a": 22}}), encoding="utf-8")  # a different size
    os.utime(path, ns=(fixed, fixed))
    assert path.stat().st_mtime_ns == fixed
    assert t.get_tunable("a", 0) == 22


def test_same_size_replacement_with_identical_mtime_ns_is_seen_through_the_inode(t, tmp_path) -> None:
    path = _file(tmp_path)
    fixed = 1_700_000_000_000_000_000
    path.write_text(json.dumps({"overrides": {"a": 1}}), encoding="utf-8")
    os.utime(path, ns=(fixed, fixed))
    assert t.get_tunable("a", 0) == 1
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"overrides": {"a": 2}}), encoding="utf-8")  # same length, new inode
    os.utime(other, ns=(fixed, fixed))
    os.replace(other, path)
    assert t.get_tunable("a", 0) == 2


def test_a_second_process_sees_a_write_within_two_seconds(t, tmp_path, repo_root) -> None:
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from brain import tunables\n"
        "print('READY', flush=True)\n"
        "end = time.monotonic() + 2.0\n"
        "while time.monotonic() < end:\n"
        "    if tunables.get_tunable('a', 1) == 5:\n"
        "        print('SEEN'); break\n"
        "    time.sleep(0.01)\n"
    )
    env = {**os.environ, "KINDLED_HOME": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"
    t.set_overrides({"a": 5})
    out, _ = proc.communicate(timeout=20)
    assert "SEEN" in out


# ---------------------------------------------------------------------------
# C36: I/O failures are not cached, content failures are
# ---------------------------------------------------------------------------


def test_an_io_failure_is_not_cached_and_the_previous_overrides_stay_in_force(t, tmp_path, monkeypatch) -> None:
    path = _file(tmp_path)
    path.write_text(json.dumps({"overrides": {"a": 1}}), encoding="utf-8")
    assert t.get_tunable("a", 0) == 1
    path.write_text(json.dumps({"overrides": {"a": 22}}), encoding="utf-8")  # a change the reader must pick up
    import brain.health.json_io as jio

    real = jio._read_bytes
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("busy")
        return real(p)

    monkeypatch.setattr(jio, "_read_bytes", flaky)
    assert t.get_tunable("a", 0) == 1  # the failed read did not wipe the overrides
    assert t.get_tunable("a", 0) == 22  # and was not cached: the next call re-read the file


def test_a_content_failure_is_cached_once_and_leaves_the_notice(t, tmp_path, monkeypatch) -> None:
    path = _file(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    import brain.tunables as mod

    real = mod.read_json_result
    reads = {"n": 0}

    def counting(*args, **kwargs):
        reads["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "read_json_result", counting)
    assert t.get_tunable("a", 5) == 5
    assert t.get_tunable("a", 5) == 5
    assert reads["n"] == 1
    notice = t.current_notice()
    assert notice is not None and notice["kind"] == "unreadable"


# ---------------------------------------------------------------------------
# Non-re-entrancy is a documented fact the callers rely on (C33, second test)
# ---------------------------------------------------------------------------


def test_set_overrides_does_not_reacquire_a_lock_its_caller_holds(t, tmp_path, monkeypatch) -> None:
    """A nested acquire of the same sidecar is NOT granted (the lock is not re-entrant): a caller holding the tunables
    lock gets a TunablesWriteError from the bounded wait instead of hanging. Nothing in the product nests."""
    from brain import dev_constants
    from brain.utils.file_lock import file_lock

    monkeypatch.setattr(dev_constants, "SETTINGS_FILE_LOCK_WAIT_S", 0.3)
    with file_lock(_file(tmp_path)):
        with pytest.raises(t.TunablesWriteError):
            t.set_overrides({"a": 1})
    assert not _file(tmp_path).exists()
