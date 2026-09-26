"""Rollback canary (#286 §7): every persisted-state reader keeps the fields it
knows and ignores fields a newer brain added. Each test writes through the
module's real writer (or its real on-disk shape), adds one unknown field, and
reads back through the real reader. See brain/state_compat.py for the rule."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

FUTURE = {"_future_field": "written by a newer brain"}


def _add_future_field(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(FUTURE)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_bridge_state(tmp_path):
    from brain.bridge import state_file
    from brain.bridge.state_file import BridgeState

    st = BridgeState(persona="p", pid=123, port=4567, started_at="2026-09-26T00:00:00Z",
                     stopped_at=None, shutdown_clean=False, client_origin="tests",
                     auth_token="tok", drain_errors=0)
    state_file.write(tmp_path, st)
    _add_future_field(tmp_path / state_file.STATE_FILENAME)
    assert state_file.read(tmp_path) == st
