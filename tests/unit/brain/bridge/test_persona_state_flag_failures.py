"""#367: the recovery-banner flags fail soft (False), but no longer silently."""

from __future__ import annotations

import logging

import pytest

import brain.bridge.persona_state as ps


@pytest.fixture(autouse=True)
def _fresh_warned(monkeypatch):
    from brain.utils import log_throttle

    monkeypatch.setattr(log_throttle, "_last", {})


def _boom(*_a, **_k):
    raise RuntimeError("detector bug")


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == ps.__name__]


def test_recovering_flag_warns_once_per_failure_kind(tmp_path, monkeypatch, caplog):
    (tmp_path / "active_conversations").mkdir()
    import brain.chat.session as session

    monkeypatch.setattr(session, "all_sessions", _boom)
    with caplog.at_level(logging.DEBUG, logger=ps.__name__):
        assert ps._is_recovering(tmp_path) is False
        assert ps._is_recovering(tmp_path) is False  # the next poll: debug only
    (w,) = _warnings(caplog)
    assert "detector bug" in w.getMessage() or w.exc_info


def test_narrative_flag_warns_on_failure(tmp_path, monkeypatch, caplog):
    import brain.narrative_memory.state as nm_state

    monkeypatch.setattr(nm_state, "load_or_recover", _boom)
    with caplog.at_level(logging.WARNING, logger=ps.__name__):
        assert ps._narrative_memory_replayed(tmp_path) is False
    assert len(_warnings(caplog)) == 1


def test_felt_time_flag_warns_on_a_corrupt_file(tmp_path, caplog):
    (tmp_path / "felt_time_state.json").write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=ps.__name__):
        assert ps._felt_time_recovered(tmp_path) is False
    assert len(_warnings(caplog)) == 1


def test_felt_time_flag_reads_utf8_regardless_of_the_locale(tmp_path, monkeypatch):
    """A non-ASCII byte in the file read as cp1252 (Windows default) used to fail."""
    (tmp_path / "felt_time_state.json").write_bytes('{"replayed": true, "note": "\u26a0\ufe0f"}'.encode())  # 0x8F: undefined in cp1252
    import pathlib

    real = pathlib.Path.read_text

    def cp1252_default(self, encoding=None, errors=None, newline=None):
        return real(self, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", cp1252_default)
    assert ps._felt_time_recovered(tmp_path) is True
