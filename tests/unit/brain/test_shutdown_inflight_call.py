"""A model call already running when the stop arrives finishes and its result is kept (C5, D2, D21).

Preservation: the stop never cancels a call in flight. The fake `claude` blocks until the test releases it; the stop is
requested meanwhile. The call returns normally, the caller persists its result (the session's extraction cursor
advances past the extracted turns), and no second call is spawned after it.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.bridge.provider import ClaudeCliProvider
from brain.ingest.buffer import ingest_turn, read_cursor, read_session
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


def _session(pd: Path, sid: str) -> None:
    for i in range(2):
        ingest_turn(pd, {"session_id": sid, "speaker": "user" if i == 0 else "assistant",
                         "text": f"{sid} turn {i}", "ts": f"2026-01-01T10:0{i}:00+00:00"})


def test_a_call_in_flight_at_the_stop_completes_and_is_persisted(tmp_path, monkeypatch):
    pd = tmp_path / "nell"
    (pd / "active_conversations").mkdir(parents=True)
    _session(pd, "sess_a")
    _session(pd, "sess_b")
    started, release = threading.Event(), threading.Event()
    spawned: list[int] = []
    items = [{"text": "the user likes rain", "label": "fact", "importance": 3, "emotions": {}}]

    def fake_run(cmd, *a, **kw):
        spawned.append(1)
        started.set()
        assert release.wait(30)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": json.dumps(items)}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    store = MemoryStore(pd / "memories.db", integrity_check=False)
    hebbian = HebbianMatrix(pd / "hebbian.db")
    outcome: dict = {}

    def sweep():
        from brain.ingest.pipeline import snapshot_stale_sessions

        try:
            outcome["reports"] = snapshot_stale_sessions(
                pd, silence_minutes=0.0, store=store, hebbian=hebbian, provider=ClaudeCliProvider(model="m")
            )
        except Exception as exc:  # noqa: BLE001 - the second session's refusal ends the sweep
            outcome["exc"] = exc

    t = threading.Thread(target=sweep, daemon=True)
    t.start()
    try:
        assert started.wait(30)
        cli_throttle.request_shutdown()
        release.set()
        t.join(30)
    finally:
        release.set()
        store.close()
        hebbian.close()
    assert not t.is_alive()
    assert len(spawned) == 1, "a second call was spawned after the stop"
    done = [sid for sid in ("sess_a", "sess_b") if read_cursor(pd, sid) is not None]
    assert len(done) == 1, "the call in flight at the stop was not persisted"
    sid = done[0]
    assert read_cursor(pd, sid) == read_session(pd, sid)[-1]["ts"]
