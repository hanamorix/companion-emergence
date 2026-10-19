"""#462: a corrupted or unreadable tunables.json must not be rebuilt with empty overrides silently.

`write_defaults_section` (bridge start) read tunables.json and rewrote it; malformed content, a non-object value, or an `overrides`
that was not an object took the empty default and the user's overrides were thrown away with one log line. Now the damaged file is
kept aside as `tunables.json.corrupt-<ts>` and the fact is PERSISTED beside the file in `tunables.notice.json`, so every bridge for
every persona sharing the home reports it (through `/persona/state`) until the user acts: edits the overrides, removes the kept file,
or (for an unreadable file) the runtime actually reloads. `current_notice()` is a pure read that never raises.

Synthetic names only; `KINDLED_HOME` is a tmp dir.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from brain.health.json_io import JsonFileError

_NOTICE = "tunables.notice.json"


@pytest.fixture()
def tunables(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield t
    t._reset_for_tests()


def _kept(home: Path) -> list[Path]:
    return sorted(home.glob("tunables.json.corrupt-*"))


def _live(home: Path) -> dict:
    return json.loads((home / "tunables.json").read_text(encoding="utf-8"))


def _marker(home: Path) -> dict:
    return json.loads((home / _NOTICE).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the damaged file is kept aside
# ---------------------------------------------------------------------------


def test_a_malformed_file_is_kept_aside_and_the_live_file_is_rebuilt(tunables, tmp_path):
    # C462-a
    original = b'{"overrides": {"x": 1}, '
    (tmp_path / "tunables.json").write_bytes(original)
    tunables.register("a.b", 60.0)

    tunables.write_defaults_section()

    kept = _kept(tmp_path)
    assert len(kept) == 1 and kept[0].read_bytes() == original
    live = _live(tmp_path)
    assert live["overrides"] == {} and live["defaults"] == {"a.b": 60.0} and "_readme" in live


@pytest.mark.parametrize("text", ['{"overrides": [1]}', "[1]"], ids=["overrides-not-an-object", "top-level-not-an-object"])
def test_a_file_whose_shape_is_wrong_is_kept_aside_too(tunables, tmp_path, text):
    # C462-b: a valid JSON file whose overrides cannot be used is not silently dropped either.
    (tmp_path / "tunables.json").write_text(text, encoding="utf-8")

    tunables.write_defaults_section()

    kept = _kept(tmp_path)
    assert len(kept) == 1 and kept[0].read_text() == text
    assert _live(tmp_path)["overrides"] == {}


def test_a_valid_file_keeps_its_overrides_and_an_absent_file_is_created(tunables, tmp_path):
    # C462-c PINS
    (tmp_path / "tunables.json").write_text(json.dumps({"overrides": {"user.key": 7}}), encoding="utf-8")
    tunables.write_defaults_section()
    assert _live(tmp_path)["overrides"] == {"user.key": 7} and not _kept(tmp_path)
    assert tunables.current_notice() is None

    (tmp_path / "tunables.json").unlink()
    tunables.write_defaults_section()
    assert _live(tmp_path)["overrides"] == {} and tunables.current_notice() is None


def test_an_empty_file_writes_no_marker(tunables, tmp_path):
    # C462-c: a zero-byte file carries no overrides, so there is nothing to tell the user.
    (tmp_path / "tunables.json").write_bytes(b"")

    tunables.write_defaults_section()

    assert not (tmp_path / _NOTICE).exists() and tunables.current_notice() is None


# ---------------------------------------------------------------------------
# the persisted marker, per heal kind
# ---------------------------------------------------------------------------


def test_a_reset_heal_writes_a_reset_marker_naming_the_kept_file(tunables, tmp_path, caplog):
    # C462-d, C462-f(log)
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")

    with caplog.at_level(logging.WARNING, logger="brain.tunables"):
        tunables.write_defaults_section()

    kept = _kept(tmp_path)[0]
    marker = _marker(tmp_path)
    assert marker["kind"] == "reset" and marker["kept_as"] == kept.name and marker["overrides_at_heal"] == {}
    assert tunables.current_notice() == {"kind": "reset", "kept_as": kept.name}
    mine = [r for r in caplog.records if r.name == "brain.tunables"]
    assert len(mine) == 1 and kept.name in mine[0].getMessage() and "not in effect" in mine[0].getMessage()


def test_a_heal_from_a_valid_backup_writes_a_restored_marker(tunables, tmp_path):
    # C462-d
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")
    (tmp_path / "tunables.json.bak1").write_text(json.dumps({"overrides": {"x": 1}}), encoding="utf-8")

    tunables.write_defaults_section()

    kept = _kept(tmp_path)[0]
    assert _marker(tmp_path)["kind"] == "restored" and _marker(tmp_path)["overrides_at_heal"] == {"x": 1}
    assert tunables.current_notice() == {"kind": "restored", "kept_as": kept.name}
    assert _live(tmp_path)["overrides"] == {"x": 1}


def test_an_unreadable_file_writes_an_unreadable_marker_and_is_left_alone(tunables, tmp_path):
    # C462-d: an I/O fault (a directory where the file should be): nothing is quarantined or rewritten, no raise.
    (tmp_path / "tunables.json").mkdir()

    tunables.write_defaults_section()

    assert _marker(tmp_path)["kind"] == "unreadable" and _marker(tmp_path)["kept_as"] is None
    assert (tmp_path / "tunables.json").is_dir() and not _kept(tmp_path)
    assert tunables.current_notice() == {"kind": "unreadable", "kept_as": None}


def test_the_marker_write_is_best_effort_and_never_skips_the_defaults_rewrite(tunables, tmp_path, monkeypatch, caplog):
    # C462-d2
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")
    real = tunables.write_json

    def write_json(path, *a, **k):
        if Path(path).name == _NOTICE:
            raise OSError("read-only home")
        return real(path, *a, **k)

    monkeypatch.setattr(tunables, "write_json", write_json)
    tunables.register("a.b", 60.0)

    with caplog.at_level(logging.WARNING, logger="brain.tunables"):
        tunables.write_defaults_section()

    assert _live(tmp_path)["defaults"] == {"a.b": 60.0}  # the defaults rewrite still happened
    assert any("notice" in r.getMessage().lower() for r in caplog.records if r.name == "brain.tunables")


def test_the_marker_write_and_delete_both_take_the_marker_lock(tunables, tmp_path, monkeypatch):
    # C462-d2
    taken: list[str] = []
    real = tunables.file_lock

    @contextmanager
    def recording(path, *a, **k):
        taken.append(Path(path).name)
        with real(path, *a, **k) as acquired:
            yield acquired

    monkeypatch.setattr(tunables, "file_lock", recording)
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")
    tunables.write_defaults_section()
    assert _NOTICE in taken

    taken.clear()
    kept = _kept(tmp_path)[0]
    kept.unlink()  # the user removed the kept file: the marker is now inactive
    tunables._drop_inactive_marker()
    assert _NOTICE in taken and not (tmp_path / _NOTICE).exists()


def test_a_delete_re_evaluates_inside_the_lock_and_spares_a_newer_marker(tunables, tmp_path, monkeypatch):
    # C462-d2: a bridge that judged an OLD marker inactive must not delete a NEWER one written meanwhile.
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")
    tunables.write_defaults_section()
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


# ---------------------------------------------------------------------------
# staying active, and clearing
# ---------------------------------------------------------------------------


def test_a_second_boot_on_the_healed_file_leaves_the_notice_active(tunables, tmp_path):
    # C462-f
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")
    tunables.write_defaults_section()
    first = (tmp_path / _NOTICE).read_bytes()

    tunables.write_defaults_section()  # a second bridge's boot rewrites the file with the same overrides

    assert tunables.current_notice() is not None
    assert (tmp_path / _NOTICE).read_bytes() == first  # neither rewritten nor deleted


def test_editing_the_overrides_clears_the_notice_and_the_next_boot_deletes_the_marker(tunables, tmp_path):
    # C462-g
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")
    tunables.write_defaults_section()
    assert tunables.current_notice() is not None

    (tmp_path / "tunables.json").write_text(json.dumps({"overrides": {"x": 2}}), encoding="utf-8")

    assert tunables.current_notice() is None
    assert (tmp_path / _NOTICE).exists()  # a pure read never deletes
    tunables.write_defaults_section()
    assert not (tmp_path / _NOTICE).exists() and _live(tmp_path)["overrides"] == {"x": 2}


def test_removing_the_kept_file_clears_the_notice(tunables, tmp_path):
    # C462-g
    (tmp_path / "tunables.json").write_bytes(b"{corrupt")
    tunables.write_defaults_section()

    _kept(tmp_path)[0].unlink()

    assert tunables.current_notice() is None


def test_an_unreadable_notice_stays_until_the_runtime_actually_reloads(tunables, tmp_path, monkeypatch):
    # C462-g: the runtime caches a failed read until the file's mtime changes, so "readable again" alone is not "reloaded".
    (tmp_path / "tunables.json").write_text(json.dumps({"overrides": {"x": 1}}), encoding="utf-8")
    real = tunables.read_json_result

    def failing(path, *a, **k):
        raise JsonFileError(path, "io", "denied")

    monkeypatch.setattr(tunables, "read_json_result", failing)
    tunables.write_defaults_section()
    monkeypatch.setattr(tunables, "read_json_result", real)
    assert _marker(tmp_path)["kind"] == "unreadable"

    assert tunables.current_notice() == {"kind": "unreadable", "kept_as": None}  # permissions fixed, mtime unchanged

    path = tmp_path / "tunables.json"
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # the file changed: the runtime reloads

    assert tunables.current_notice() is None


def test_the_next_boots_clean_read_drops_an_unreadable_marker(tunables, tmp_path, monkeypatch):
    # C462-g
    (tmp_path / "tunables.json").write_text(json.dumps({"overrides": {"x": 1}}), encoding="utf-8")
    real = tunables.read_json_result
    monkeypatch.setattr(tunables, "read_json_result", lambda p, *a, **k: (_ for _ in ()).throw(JsonFileError(p, "io", "x")))
    tunables.write_defaults_section()
    monkeypatch.setattr(tunables, "read_json_result", real)

    tunables.write_defaults_section()

    assert not (tmp_path / _NOTICE).exists() and tunables.current_notice() is None


# ---------------------------------------------------------------------------
# the reporter is a pure read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", ["corrupt", "directory", "list"])
def test_current_notice_on_a_bad_marker_returns_none_and_writes_nothing(tunables, tmp_path, make):
    # C462-h
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
    # C462-h
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "does-not-exist"))

    assert tunables.current_notice() is None
    assert not (tmp_path / "does-not-exist").exists()


def test_the_overrides_cache_reset_does_not_clear_the_registry(tunables):
    # C462-j
    tunables.register("a.b", 1.0)

    tunables._reset_overrides_cache()

    assert tunables._registry["a.b"] == 1.0


# ---------------------------------------------------------------------------
# O2: the runtime reload breaking mid-run on a hand edit raises the same notice
# ---------------------------------------------------------------------------


def _touch_later(path: Path, seconds: int) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def test_a_hand_edit_that_breaks_the_file_mid_run_raises_the_unreadable_notice(tunables, tmp_path):
    path = tmp_path / "tunables.json"
    path.write_text(json.dumps({"overrides": {"a.b": 180.0}}), encoding="utf-8")
    assert tunables.get_tunable("a.b", 60.0) == 180.0
    assert tunables.current_notice() is None

    path.write_bytes(b"{broken")  # the user's hand edit, while the bridge runs
    _touch_later(path, 5)

    assert tunables.get_tunable("a.b", 60.0) == 60.0  # fail-open: code defaults
    assert tunables.current_notice() == {"kind": "unreadable", "kept_as": None}
    assert path.read_bytes() == b"{broken" and not _kept(tmp_path)  # the runtime never heals or rewrites the file

    path.write_text(json.dumps({"overrides": {"a.b": 200.0}}), encoding="utf-8")  # the user fixes it
    _touch_later(path, 10)

    assert tunables.get_tunable("a.b", 60.0) == 200.0  # reloaded by the mtime change
    assert tunables.current_notice() is None


def test_an_overrides_value_that_is_not_an_object_mid_run_raises_the_notice(tunables, tmp_path):
    path = tmp_path / "tunables.json"
    path.write_text(json.dumps({"overrides": {"a.b": 180.0}}), encoding="utf-8")
    tunables.get_tunable("a.b", 60.0)

    path.write_text(json.dumps({"overrides": [1]}), encoding="utf-8")
    _touch_later(path, 5)

    assert tunables.get_tunable("a.b", 60.0) == 60.0
    assert tunables.current_notice() == {"kind": "unreadable", "kept_as": None}


def test_the_runtime_marker_is_written_once_per_file_change_not_on_every_read(tunables, tmp_path):
    path = tmp_path / "tunables.json"
    path.write_bytes(b"{broken")
    tunables.get_tunable("a.b", 60.0)
    marker = tmp_path / _NOTICE
    first = (marker.read_bytes(), marker.stat().st_mtime_ns)

    for _ in range(5):
        tunables.get_tunable("a.b", 60.0)

    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == first


def test_a_valid_or_empty_or_absent_file_writes_no_runtime_marker(tunables, tmp_path):
    path = tmp_path / "tunables.json"
    tunables.get_tunable("a.b", 60.0)  # absent
    path.write_bytes(b"")
    _touch_later(path, 2)
    tunables.get_tunable("a.b", 60.0)  # zero-byte: no overrides to lose
    path.write_text(json.dumps({"overrides": {"a.b": 1.0}}), encoding="utf-8")
    _touch_later(path, 4)
    tunables.get_tunable("a.b", 60.0)

    assert not (tmp_path / _NOTICE).exists()


def test_the_runtime_marker_write_failing_never_breaks_a_tunable_read(tunables, tmp_path, monkeypatch):
    real = tunables.write_json
    monkeypatch.setattr(tunables, "write_json", lambda p, *a, **k: (_ for _ in ()).throw(OSError("read-only")) if Path(p).name == _NOTICE else real(p, *a, **k))
    (tmp_path / "tunables.json").write_bytes(b"{broken")

    assert tunables.get_tunable("a.b", 60.0) == 60.0


def test_a_marker_write_that_hits_a_programming_error_is_not_swallowed(tunables, tmp_path, monkeypatch):
    # N4: the best-effort marker write catches the filesystem's OSError only, so a bug in building the record shows.
    monkeypatch.setattr(tunables, "write_json", lambda *a, **k: (_ for _ in ()).throw(TypeError("a bug")))

    with pytest.raises(TypeError):
        tunables._write_marker({"kind": "reset"})
