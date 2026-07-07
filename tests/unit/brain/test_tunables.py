"""Canaries for the ops-tunables invariants (spec 2026-07-04-ops-tunables-design §5).

Fail-open, hot-reload, and type-guard are load-bearing: a regression here
silently strips or corrupts operational overrides on live installs.
"""
from __future__ import annotations

import json
import os
import time

import pytest


@pytest.fixture()
def tunables(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield t
    t._reset_for_tests()


def _write(tmp_home, overrides):
    (tmp_home / "tunables.json").write_text(
        json.dumps({"defaults": {}, "overrides": overrides}), encoding="utf-8"
    )


def test_missing_file_returns_default(tunables):
    assert tunables.get_tunable("a.b", 60.0) == 60.0


def test_override_wins(tunables, tmp_path):
    _write(tmp_path, {"a.b": 180.0})
    assert tunables.get_tunable("a.b", 60.0) == 180.0


def test_register_returns_default_and_records(tunables):
    assert tunables.register("a.c", 15.0) == 15.0
    assert tunables._registry["a.c"] == 15.0


def test_fails_open_on_corrupt_json(tunables, tmp_path):
    (tmp_path / "tunables.json").write_text("{not json", encoding="utf-8")
    assert tunables.get_tunable("a.b", 60.0) == 60.0  # no raise


def test_type_guard_ignores_wrong_type(tunables, tmp_path):
    _write(tmp_path, {"a.b": "fast"})
    assert tunables.get_tunable("a.b", 60.0) == 60.0


def test_type_guard_accepts_int_for_float(tunables, tmp_path):
    _write(tmp_path, {"a.b": 90})
    val = tunables.get_tunable("a.b", 60.0)
    assert val == 90.0 and isinstance(val, float)


def test_type_guard_bool_is_not_int(tunables, tmp_path):
    _write(tmp_path, {"a.n": True})
    assert tunables.get_tunable("a.n", 5) == 5  # bool must not satisfy int


def test_hot_reload_on_mtime_change(tunables, tmp_path):
    _write(tmp_path, {"a.b": 90.0})
    assert tunables.get_tunable("a.b", 60.0) == 90.0
    _write(tmp_path, {"a.b": 240.0})
    # ensure mtime actually advances on coarse-granularity filesystems
    os.utime(tmp_path / "tunables.json", (time.time() + 2, time.time() + 2))
    assert tunables.get_tunable("a.b", 60.0) == 240.0


def test_write_defaults_creates_file(tunables, tmp_path):
    tunables.register("a.b", 60.0)
    tunables.write_defaults_section()
    data = json.loads((tmp_path / "tunables.json").read_text(encoding="utf-8"))
    assert data["defaults"] == {"a.b": 60.0}
    assert data["overrides"] == {}
    assert "_readme" in data


def test_write_defaults_refreshes_stale_default_preserves_overrides(tunables, tmp_path):
    # Simulates a code-default change on upgrade: file has old default 60.0,
    # code now registers 180.0. Boot must refresh defaults, keep overrides.
    (tmp_path / "tunables.json").write_text(
        json.dumps({"defaults": {"a.b": 60.0}, "overrides": {"user.key": 7}}),
        encoding="utf-8",
    )
    tunables.register("a.b", 180.0)
    tunables.write_defaults_section()
    data = json.loads((tmp_path / "tunables.json").read_text(encoding="utf-8"))
    assert data["defaults"] == {"a.b": 180.0}
    assert data["overrides"] == {"user.key": 7}


def test_write_defaults_survives_corrupt_file(tunables, tmp_path):
    (tmp_path / "tunables.json").write_text("{corrupt", encoding="utf-8")
    tunables.register("a.b", 60.0)
    tunables.write_defaults_section()  # must not raise; rebuilds file, empty overrides
    data = json.loads((tmp_path / "tunables.json").read_text(encoding="utf-8"))
    assert data["defaults"] == {"a.b": 60.0}
    assert data["overrides"] == {}
