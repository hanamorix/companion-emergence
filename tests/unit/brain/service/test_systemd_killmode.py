"""systemd `KillMode=mixed` (C47, D14): built into new units, and added idempotently, with one log line, to OUR
already-installed unit only."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from brain.service import systemd

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="systemd unit text uses POSIX paths")


@pytest.fixture
def unit_env(tmp_path, monkeypatch):
    nell = tmp_path / "bin" / "nell"
    nell.parent.mkdir()
    nell.write_text("#!/bin/sh\n")
    nell.chmod(0o755)
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    monkeypatch.setattr(systemd, "user_unit_dir", lambda: unit_dir)
    monkeypatch.setattr(systemd, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    calls: list[list[str]] = []
    monkeypatch.setattr(systemd, "run_systemctl", lambda args: calls.append(list(args)) or None)
    return nell, unit_dir, calls


def test_built_unit_has_killmode_mixed(unit_env):
    nell, _, _ = unit_env
    body = systemd.build_systemd_unit_text(persona="nell", nell_path=nell)
    assert "KillMode=mixed\n" in body
    assert body.index("TimeoutStopSec=300") < body.index("KillMode=mixed")


def _old_unit(nell: Path) -> str:
    body = systemd.build_systemd_unit_text(persona="nell", nell_path=nell)
    return body.replace("KillMode=mixed\n", "")


def test_installed_unit_without_killmode_is_refreshed_once(unit_env, caplog):
    nell, unit_dir, calls = unit_env
    path = unit_dir / systemd.service_unit_name("nell")
    path.write_text(_old_unit(nell), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="brain.service.systemd"):
        assert systemd.refresh_installed_unit("nell") is True
    text = path.read_text(encoding="utf-8")
    assert "TimeoutStopSec=300\nKillMode=mixed\n" in text
    assert calls == [["daemon-reload"]]
    assert sum("KillMode=mixed" in r.getMessage() for r in caplog.records) == 1
    # idempotent: a second run changes nothing and logs nothing
    caplog.clear()
    before = path.read_bytes()
    with caplog.at_level(logging.INFO, logger="brain.service.systemd"):
        assert systemd.refresh_installed_unit("nell") is False
    assert path.read_bytes() == before
    assert calls == [["daemon-reload"]]
    assert not [r for r in caplog.records if "KillMode" in r.getMessage()]


def test_a_unit_we_did_not_write_is_left_alone(unit_env):
    _, unit_dir, calls = unit_env
    path = unit_dir / systemd.service_unit_name("nell")
    foreign = "[Unit]\nDescription=someone else's unit\n[Service]\nExecStart=/bin/true\nTimeoutStopSec=300\n"
    path.write_text(foreign, encoding="utf-8")
    assert systemd.refresh_installed_unit("nell") is False
    assert path.read_text(encoding="utf-8") == foreign
    assert calls == []


def test_no_installed_unit_is_a_noop(unit_env):
    _, _, calls = unit_env
    assert systemd.refresh_installed_unit("nell") is False
    assert calls == []
