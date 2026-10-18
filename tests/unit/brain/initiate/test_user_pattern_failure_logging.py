"""#367: presence derivations fail soft, but no longer below the runtime log's
INFO level: each kind warns once (with traceback), then drops to debug."""

from __future__ import annotations

import logging

import brain.initiate.user_pattern as up


def test_a_failing_presence_load_warns_once(tmp_path, monkeypatch, caplog):
    import brain.initiate.presence_state as presence_state
    from brain.utils import log_throttle

    monkeypatch.setattr(log_throttle, "_last", {})

    def boom(*_a, **_k):
        raise RuntimeError("presence file unreadable")

    monkeypatch.setattr(presence_state, "load_presence_state", boom)
    with caplog.at_level(logging.DEBUG, logger=up.__name__):
        up.compute_user_presence(tmp_path)
        up.compute_user_presence(tmp_path)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "load_presence_state" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].exc_info
