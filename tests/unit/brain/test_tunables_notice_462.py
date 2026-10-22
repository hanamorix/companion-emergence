"""#462 reshaped (#529, owner ruling E9): an unreadable ``tunables.toml`` is LEFT ALONE and the notice names the line.

The file is edited by hand, so the app never quarantines, resets or repairs it. While it does not parse, every tunable reads
its code default, and the persisted notice (``tunables.notice.json``, still JSON) says which line and column failed so the
user can fix that line. Every bridge sharing the home reports the same marker. ``current_notice()`` is a pure read that
never raises. Synthetic names only; ``KINDLED_HOME`` is a tmp dir.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

_NOTICE = "tunables.notice.json"
_FILE = "tunables.toml"


@pytest.fixture()
def tunables(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield t
    t._reset_for_tests()


def _marker(home: Path) -> dict:
    return json.loads((home / _NOTICE).read_text(encoding="utf-8"))


def _touch_later(path: Path, seconds: int) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def test_the_marker_records_kind_line_column_reason_and_the_mtime(tunables, tmp_path):
    (tmp_path / _FILE).write_text("a = 1\nb = = 2\n", encoding="utf-8")
    tunables.sync_file()
    m = _marker(tmp_path)
    assert m["kind"] == "unreadable" and m["line"] == 2 and isinstance(m["column"], int) and m["reason"]
    assert m["mtime_ns"] == (tmp_path / _FILE).stat().st_mtime_ns
    assert "at" in m
    assert tunables.current_notice() == {"kind": "unreadable", "line": 2, "column": m["column"], "reason": m["reason"]}


def test_a_log_warning_names_the_file_and_the_line_once(tunables, tmp_path, caplog):
    (tmp_path / _FILE).write_text("a = 1\nb = = 2\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brain.tunables"):
        tunables.sync_file()
        tunables.get_tunable("a", 1)
        tunables.get_tunable("a", 1)
    texts = [r.getMessage() for r in caplog.records if r.name == "brain.tunables" and "line 2" in r.getMessage()]
    assert texts and all(_FILE in x for x in texts)
    assert len(texts) <= 2  # once from the boot pass, once from the first runtime read; never once per read


def test_the_marker_write_is_best_effort_and_never_breaks_the_boot_pass(tunables, tmp_path, monkeypatch, caplog):
    (tmp_path / _FILE).write_text("a = = 1\n", encoding="utf-8")
    real = tunables.write_json

    def write_json(path, *a, **k):
        if Path(path).name == _NOTICE:
            raise OSError("read-only home")
        return real(path, *a, **k)

    monkeypatch.setattr(tunables, "write_json", write_json)
    with caplog.at_level(logging.WARNING, logger="brain.tunables"):
        tunables.sync_file()
    assert any("notice" in r.getMessage().lower() for r in caplog.records if r.name == "brain.tunables")


def test_the_marker_write_and_delete_both_take_the_marker_lock(tunables, tmp_path, monkeypatch):
    taken: list[str] = []
    real = tunables.file_lock

    @contextmanager
    def recording(path, *a, **k):
        taken.append(Path(path).name)
        with real(path, *a, **k) as acquired:
            yield acquired

    monkeypatch.setattr(tunables, "file_lock", recording)
    path = tmp_path / _FILE
    path.write_text("a = = 1\n", encoding="utf-8")
    tunables.sync_file()
    assert _NOTICE in taken

    taken.clear()
    path.write_text("a = 1\n", encoding="utf-8")  # the user fixed the file: the marker is now inactive
    _touch_later(path, 5)
    tunables._drop_inactive_marker()
    assert _NOTICE in taken and not (tmp_path / _NOTICE).exists()


def test_a_delete_re_evaluates_inside_the_lock_and_spares_a_newer_marker(tunables, tmp_path, monkeypatch):
    (tmp_path / _FILE).write_text("a = = 1\n", encoding="utf-8")
    tunables.sync_file()
    real = tunables._marker_is_active
    calls = {"n": 0}

    def fake(marker):
        calls["n"] += 1
        if calls["n"] == 1:
            return False  # the first (outside-the-lock) judgement: "inactive"
        return real(marker)  # inside the lock the marker is judged again, for real

    monkeypatch.setattr(tunables, "_marker_is_active", fake)
    tunables._drop_inactive_marker()
    assert calls["n"] >= 2 and (tmp_path / _NOTICE).exists()


def test_a_second_boot_on_the_still_broken_file_leaves_the_notice_active(tunables, tmp_path):
    (tmp_path / _FILE).write_text("a = = 1\n", encoding="utf-8")
    tunables.sync_file()
    tunables.sync_file()
    assert tunables.current_notice() is not None
    assert (tmp_path / _FILE).read_text(encoding="utf-8") == "a = = 1\n"


def test_fixing_the_file_clears_the_notice_and_the_next_boot_deletes_the_marker(tunables, tmp_path):
    path = tmp_path / _FILE
    path.write_text("a = = 1\n", encoding="utf-8")
    tunables.sync_file()
    path.write_text("a = 1\n", encoding="utf-8")
    _touch_later(path, 5)
    assert tunables.current_notice() is None
    assert (tmp_path / _NOTICE).exists()  # a pure read never deletes
    tunables.sync_file()
    assert not (tmp_path / _NOTICE).exists()


def test_an_io_unreadable_notice_stays_until_the_runtime_actually_reloads(tunables, tmp_path, monkeypatch):
    path = tmp_path / _FILE
    path.write_text('"a.b" = 1.0\n', encoding="utf-8")
    from brain.utils import toml_io

    real = toml_io._read_bytes

    def boom(p):
        raise PermissionError("busy")

    monkeypatch.setattr(toml_io, "_read_bytes", boom)
    assert tunables.get_tunable("a.b", 60.0) == 60.0
    notice = tunables.current_notice()
    assert notice is not None and notice["kind"] == "unreadable" and notice["line"] is None
    monkeypatch.setattr(toml_io, "_read_bytes", real)
    assert tunables.current_notice() is not None  # the mtime did not change: still active
    _touch_later(path, 5)
    assert tunables.get_tunable("a.b", 60.0) == 1.0
    assert tunables.current_notice() is None


@pytest.mark.parametrize("make", ["corrupt", "directory", "list"])
def test_current_notice_on_a_bad_marker_returns_none_and_writes_nothing(tunables, tmp_path, make):
    marker = tmp_path / _NOTICE
    if make == "corrupt":
        marker.write_bytes(b"{nope")
    elif make == "directory":
        marker.mkdir()
    else:
        marker.write_text("[1]")
    before = sorted(p.name for p in tmp_path.iterdir())
    assert tunables.current_notice() is None
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_current_notice_with_a_missing_home_returns_none(tunables, tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "does-not-exist"))
    assert tunables.current_notice() is None
    assert not (tmp_path / "does-not-exist").exists()


def test_the_cache_reset_does_not_clear_the_registry(tunables):
    tunables.register("a.b", 1.0)
    tunables._reset_values_cache()
    assert tunables._registry["a.b"] == 1.0


# O2: the runtime reload breaking mid-run on a hand edit raises the same notice


def test_a_hand_edit_that_breaks_the_file_mid_run_raises_the_notice_and_the_file_is_never_touched(tunables, tmp_path):
    path = tmp_path / _FILE
    path.write_text('"a.b" = 180.0\n', encoding="utf-8")
    assert tunables.get_tunable("a.b", 60.0) == 180.0
    assert tunables.current_notice() is None

    path.write_bytes(b'"a.b" = 180.0\nbroken line here\n')  # the user's hand edit, while the bridge runs
    _touch_later(path, 5)
    assert tunables.get_tunable("a.b", 60.0) == 60.0  # fail-open: code defaults
    notice = tunables.current_notice()
    assert notice is not None and notice["kind"] == "unreadable" and notice["line"] == 2
    assert path.read_bytes() == b'"a.b" = 180.0\nbroken line here\n'  # the runtime never heals or rewrites the file
    assert not [p for p in tmp_path.iterdir() if "corrupt" in p.name or "bak" in p.name]

    path.write_text('"a.b" = 200.0\n', encoding="utf-8")  # the user fixes it
    _touch_later(path, 10)
    assert tunables.get_tunable("a.b", 60.0) == 200.0
    assert tunables.current_notice() is None


def test_the_runtime_marker_is_written_once_per_file_change_not_on_every_read(tunables, tmp_path):
    (tmp_path / _FILE).write_bytes(b"a = = 1\n")
    tunables.get_tunable("a.b", 60.0)
    marker = tmp_path / _NOTICE
    first = (marker.read_bytes(), marker.stat().st_mtime_ns)
    for _ in range(5):
        tunables.get_tunable("a.b", 60.0)
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == first


def test_a_valid_or_empty_or_absent_file_writes_no_runtime_marker(tunables, tmp_path):
    path = tmp_path / _FILE
    tunables.get_tunable("a.b", 60.0)  # absent
    path.write_bytes(b"")
    _touch_later(path, 2)
    tunables.get_tunable("a.b", 60.0)  # zero-byte: nothing of the user's in it
    path.write_text('"a.b" = 1.0\n', encoding="utf-8")
    _touch_later(path, 4)
    tunables.get_tunable("a.b", 60.0)
    assert not (tmp_path / _NOTICE).exists()


def test_the_runtime_marker_write_failing_never_breaks_a_tunable_read(tunables, tmp_path, monkeypatch):
    real = tunables.write_json
    monkeypatch.setattr(
        tunables,
        "write_json",
        lambda p, *a, **k: (_ for _ in ()).throw(OSError("read-only")) if Path(p).name == _NOTICE else real(p, *a, **k),
    )
    (tmp_path / _FILE).write_bytes(b"a = = 1\n")
    assert tunables.get_tunable("a.b", 60.0) == 60.0


def test_a_marker_write_that_hits_a_programming_error_is_not_swallowed(tunables, tmp_path, monkeypatch):
    # N4: the best-effort marker write catches the filesystem's OSError only, so a bug in building the record shows.
    monkeypatch.setattr(tunables, "write_json", lambda *a, **k: (_ for _ in ()).throw(TypeError("a bug")))
    with pytest.raises(TypeError):
        tunables._write_marker({"kind": "unreadable"})


def test_the_names_of_the_removed_heal_notices_are_gone(tunables):
    for gone in ("_record_heal", "quarantine_damaged"):
        assert not hasattr(tunables, gone)
    marker = Path(tunables._notice_path())
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"kind": "reset", "kept_as": "x"}), encoding="utf-8")
    assert tunables.current_notice() is None  # an old marker kind is never reported
