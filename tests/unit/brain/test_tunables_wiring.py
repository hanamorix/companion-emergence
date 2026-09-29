"""Wiring canaries: each ops-tier call site actually consults tunables."""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tunables_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield tmp_path
    t._reset_for_tests()


def _write(home, overrides):
    (home / "tunables.json").write_text(
        json.dumps({"defaults": {}, "overrides": overrides}), encoding="utf-8"
    )


def test_chat_idle_lull_override(tunables_home):
    """The ONE registered ops-timing key (ram-spike-fix INC-6, C4(b)/C27):
    chat.idle_lull_seconds, read only by cli_throttle's private
    _lull_seconds(), called only from is_chat_idle()."""
    _write(tunables_home, {"chat.idle_lull_seconds": 30.0})
    from brain.bridge import cli_throttle

    assert cli_throttle._lull_seconds() == 30.0


def test_throttle_concurrency_override(tunables_home):
    _write(tunables_home, {"throttle.max_concurrent_background": 2})
    from brain.bridge import cli_throttle

    assert cli_throttle._max_concurrent_background() == 2


def test_keepalive_override(tunables_home):
    _write(tunables_home, {"bridge.stream_keepalive_seconds": 5.0})
    from brain.bridge import server

    assert server._stream_keepalive_seconds() == 5.0


def test_read_file_cap_override(tunables_home):
    _write(tunables_home, {"files.read_max_bytes": 1024})
    from brain.tools.impls import read_file

    assert read_file._file_read_max_bytes() == 1024
