"""One golden fixture set for the watcher files and `status --json` (C48).

tests/fixtures/bridge_watcher_files/ holds the progress record, the hard-exit request, the stop intent (pending and
committed) and `nell supervisor status --json`. The Python writers' output must equal them; the Rust and TypeScript
readers parse the same files in CI. A format change is made here and in every reader together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.bridge import daemon, shutdown_progress, state_file

FIX = Path(__file__).resolve().parents[3] / "fixtures" / "bridge_watcher_files"
TOKEN = "linux:0b9c2f4e-1111-2222-3333-444455556666:4242:123456"


def _fixture(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_progress_record_matches_the_fixture(tmp_path):
    rec = shutdown_progress.progress_record(
        pid=4242, pid_token=TOKEN, persona="nell", requested_at="2026-10-19T21:00:00.000000+00:00",
        reason="user_stop", tasks=["background job: compaction", "chat reply"],
        updated_at="2026-10-19T21:00:01.000000+00:00",
    )
    assert rec == _fixture("progress.json")
    shutdown_progress.write_record(tmp_path, shutdown_progress.PROGRESS_FILENAME, rec)
    assert shutdown_progress.read_record(tmp_path, shutdown_progress.PROGRESS_FILENAME) == _fixture("progress.json")


def test_hard_exit_request_matches_the_fixture():
    rec = shutdown_progress.hard_exit_request(pid=4242, pid_token=TOKEN,
                                              requested_at="2026-10-19T21:03:00.000000+00:00")
    assert rec == _fixture("hard_exit_request.json")


@pytest.mark.parametrize("state", ["pending", "committed"])
def test_stop_intent_matches_the_fixture(state):
    rec = shutdown_progress.stop_intent(pid=4242, pid_token=TOKEN, state=state, nonce="a1b2c3d4e5f60718",
                                        at="2026-10-19T21:04:00.000000+00:00",
                                        boot="0b9c2f4e-1111-2222-3333-444455556666", legacy=False,
                                        helper_pid=5151,
                                        helper_token="linux:0b9c2f4e-1111-2222-3333-444455556666:5151:654321")
    assert rec == _fixture(f"stop_intent_{state}.json")


def test_status_json_matches_the_fixture(tmp_path, monkeypatch):
    pd = tmp_path / "nell"
    pd.mkdir()
    state_file.write(pd, state_file.BridgeState(
        persona="nell", pid=4242, port=5000, started_at="t", stopped_at=None, shutdown_clean=False,
        client_origin="tauri", pid_token=TOKEN, identity="token", phase="stopping", stop_reason="user_stop",
        stop_requested_at="2026-10-19T21:00:00.000000+00:00",
        supervisor={"state": "running", "restarts": 0, "last_error": None, "last_error_at": None}))
    shutdown_progress.write_record(pd, shutdown_progress.PROGRESS_FILENAME, _fixture("progress.json"))
    shutdown_progress.write_record(pd, shutdown_progress.INTENT_FILENAME, _fixture("stop_intent_pending.json"))
    monkeypatch.setattr(state_file, "bridge_liveness", lambda st, p: state_file.ALIVE)
    assert daemon.status_dict(pd) == _fixture("status.json")


def test_readers_ignore_a_record_of_another_bridge(tmp_path, monkeypatch):
    """C43 (a): the progress record of an earlier process is not this bridge's (no `remaining` in status)."""
    pd = tmp_path / "nell"
    pd.mkdir()
    state_file.write(pd, state_file.BridgeState(
        persona="nell", pid=4242, port=5000, started_at="t", stopped_at=None, shutdown_clean=False,
        client_origin="tauri", pid_token="linux:other:4242:9", identity="token", phase="running"))
    shutdown_progress.write_record(pd, shutdown_progress.PROGRESS_FILENAME, _fixture("progress.json"))
    monkeypatch.setattr(state_file, "bridge_liveness", lambda st, p: state_file.ALIVE)
    assert daemon.status_dict(pd)["remaining"] is None
