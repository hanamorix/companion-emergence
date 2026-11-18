"""Hana's 11-18 review of #633, point 5 (owner S112): `/persona/state` reports the persisted rebuild notice of its own persona.

The notice (`rebuild_notice.json`, written by `engine_files.settle_rebuild_notice`) says why a switch-over did not complete and the
bridge started on the state saved before. The state carries `{"reason", "log_path"}` for the persona that has one and None for one
that has not; polling only reads. Synthetic names only; `KINDLED_HOME` is a tmp dir.
"""

from __future__ import annotations

import json

import pytest

from brain import paths
from brain.bridge import persona_state
from brain.bridge.persona_state import build_persona_state
from brain.emotion import engine_files


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield tmp_path
    t._reset_for_tests()


def _personas(home):
    out = []
    for name in ("canary", "wren"):
        d = home / "personas" / name
        d.mkdir(parents=True)
        out.append(d)
    return out


def _plant(persona_dir, reason="the first rebuild failed (OSError: disk)"):
    engine_files.rebuild_notice_path(persona_dir).write_text(json.dumps({"at": "2026-11-18T12:00:00Z", "reason": reason}), encoding="utf-8")


def _notice(persona_dir):
    return build_persona_state(persona_dir)["rebuild_notice"]


def test_a_persona_with_no_record_reports_none_for_both_personas(home):
    p1, p2 = _personas(home)
    assert _notice(p1) is None and _notice(p2) is None


def test_a_record_is_reported_for_its_own_persona_only_with_the_reason_and_the_runtime_log_path(home):
    p1, p2 = _personas(home)
    _plant(p1)
    notice = _notice(p1)
    assert set(notice) == {"reason", "log_path"}
    assert notice["reason"] == "the first rebuild failed (OSError: disk)"
    assert notice["log_path"] == str(paths.runtime_log_path("canary"))
    assert _notice(p2) is None


def test_polling_never_writes_or_deletes_the_record(home):
    (p1,) = _personas(home)[:1]
    _plant(p1)
    path = engine_files.rebuild_notice_path(p1)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert _notice(p1) is not None and _notice(p1) is not None
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_a_record_without_a_text_reason_reports_none(home):
    (p1,) = _personas(home)[:1]
    engine_files.rebuild_notice_path(p1).write_text('{"at": "2026-11-18T12:00:00Z"}', encoding="utf-8")
    assert _notice(p1) is None


def test_a_log_path_that_cannot_be_worked_out_still_sends_the_reason_with_a_null_path(home, monkeypatch):
    (p1,) = _personas(home)[:1]
    _plant(p1)

    def boom(name):
        raise RuntimeError("no log folder")

    monkeypatch.setattr(paths, "runtime_log_path", boom)
    assert _notice(p1) == {"reason": "the first rebuild failed (OSError: disk)", "log_path": None}


def test_a_failed_read_reports_none_and_the_state_is_still_built(home, monkeypatch):
    (p1,) = _personas(home)[:1]
    _plant(p1)

    def boom(persona_dir):
        raise RuntimeError("read blew up")

    monkeypatch.setattr(engine_files, "read_rebuild_notice", boom)
    assert _notice(p1) is None


def test_the_key_is_just_before_the_tunables_notice_which_stays_last(home):
    (p1,) = _personas(home)[:1]
    keys = list(persona_state.build_persona_state(p1))
    assert keys[-2:] == ["rebuild_notice", "tunables_notice"]
