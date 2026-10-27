"""Real-process stop tests with model calls in flight (#456: C14 b and f end to end, C52, C59, C64's `claude` half).

The runner's `claude` is the fake one on PATH (tests/bridge/fake_claude/claude): it starts a grandchild named like the
MCP server and blocks until the test releases it. KINDLED_TEST_CALLS starts real calls inside the bridge
(tests/bridge/_stop_hooks.py): a compaction fold, a pass-2 extraction and a soul review.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil
import pytest

from brain.bridge import shutdown_progress
from tests.bridge.test_shutdown_e2e import Bridge, _until

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows legs run in CI")


@pytest.fixture
def bridges():
    started: list[Bridge] = []
    yield started
    for b in started:
        b.release_all(8)
        b.kill()

CALLS = {
    "compaction": "tests.bridge._stop_hooks:compaction_call",
    "pass2": "tests.bridge._stop_hooks:pass2_call",
    "soul": "tests.bridge._stop_hooks:soul_call",
    "generate": "tests.bridge._stop_hooks:generate_call",
}


def _prepare(pd: Path) -> None:
    from brain.chat import pass2_queue
    from brain.ingest.buffer import ingest_turn, read_session, write_cursor

    start = datetime.now(UTC) - timedelta(days=4)
    for i in range(6):
        ingest_turn(pd, {"session_id": "sess_c", "speaker": "user" if i % 2 == 0 else "assistant",
                         "text": f"turn {i} about the rain", "ts": (start + timedelta(minutes=i)).isoformat()})
    write_cursor(pd, "sess_c", read_session(pd, "sess_c")[-1]["ts"])
    pass2_queue.enqueue({"id": pass2_queue.new_record_id(), "kind": "monologue",
                         "monologue_text": "I keep thinking about the rain.", "visible_reply": "It rained.",
                         "recent_user_msgs": ["how was it"]}, persona_dir=pd)
    (pd / "soul_candidates.jsonl").write_text(json.dumps({
        "id": "cand-0", "text": "a meaningful moment", "label": "test", "importance": 8.0,
        "queued_at": "2026-01-01T00:00:00+00:00", "source": "test", "status": "auto_pending"}) + "\n",
        encoding="utf-8")


def _snapshot(pd: Path) -> dict[str, bytes]:
    names = ["active_conversations/sess_c.jsonl", "pass2_queue.json", "soul_candidates.jsonl"]
    out = {}
    for n in names:
        f = pd / n
        out[n] = f.read_bytes() if f.exists() else b""
    return out


def _wait_calls(b: Bridge, n: int) -> list[int]:
    return _until(lambda: len(list(b.fake_dir.glob("pids-*"))) >= n and b.fake_pids(), timeout=60)


def _all_gone(pids, timeout=15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                    alive.append(pid)
            except psutil.NoSuchProcess:
                pass
        if not alive:
            return True
        time.sleep(0.1)
    return False


def test_hard_exit_with_calls_in_flight_stores_nothing_and_leaves_no_descendant(tmp_path, bridges):
    """C52 (hard-exit leg) and C14 (b): the fake claude children and their MCP-named grandchildren are gone; no
    compaction stand-in, the pass-2 record is still queued, no soul-review failure counted or audited."""
    b = Bridge(tmp_path, calls=(CALLS["compaction"], CALLS["pass2"], CALLS["soul"]), provider="claude-cli",
               prepare=_prepare)
    bridges.append(b)
    pids = _wait_calls(b, 3)
    before = _snapshot(b.pd)
    assert b.post("/supervisor/shutdown", {"reason": "user_stop"}).status_code == 202
    shutdown_progress.write_record(b.pd, shutdown_progress.REQUEST_FILENAME,
                                   {"pid": b.proc.pid, "pid_token": b.state.pid_token})
    assert b.wait_exit(30) == 0
    assert _all_gone(pids), "a claude child or MCP grandchild survived the hard exit"
    assert _snapshot(b.pd) == before
    assert not (b.pd / "soul_audit.jsonl").exists() or "model call failed" not in (
        b.pd / "soul_audit.jsonl").read_text(encoding="utf-8")
    st = b.record()
    assert st.phase == "hard_exit" and not st.shutdown_clean


@pytest.mark.slow(reason='2.3 s on macos in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups")
def test_sigint_to_the_process_group_leaves_the_claude_call_to_finish(tmp_path, bridges, monkeypatch):
    """C59 (D23): SIGINT to the runner's whole process group during an in-flight call: the fake claude is not killed,
    the call completes and its result is persisted, and the stop is a normal first-signal stop."""
    b = Bridge(tmp_path, calls=(CALLS["generate"],), provider="claude-cli", new_group=True)
    bridges.append(b)
    pids = _wait_calls(b, 1)
    os.killpg(b.proc.pid, signal.SIGINT)
    _until(lambda: (st := b.record()) and st.phase == "stopping")
    time.sleep(0.5)
    assert all(psutil.Process(p).is_running() for p in pids), "the claude call was killed by the group's SIGINT"
    b.fake_release.write_text("go")
    assert b.wait_exit(60) == 0
    assert (b.pd / "hook_generate_result.txt").read_text(encoding="utf-8") == "ok"
    st = b.record()
    assert st.phase == "stopped" and st.shutdown_clean and st.stop_reason == "os_signal"


@pytest.mark.slow(reason='2.7 s on macos in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_sighup_leaves_the_claude_call_to_finish(tmp_path, bridges):
    """C64 (`claude` half): SIGHUP starts a graceful stop and does not kill the fake claude child."""
    b = Bridge(tmp_path, calls=(CALLS["generate"],), provider="claude-cli")
    bridges.append(b)
    pids = _wait_calls(b, 1)
    b.proc.send_signal(signal.SIGHUP)
    _until(lambda: (st := b.record()) and st.phase == "stopping")
    time.sleep(0.5)
    assert all(psutil.Process(p).is_running() for p in pids)
    b.fake_release.write_text("go")
    assert b.wait_exit(60) == 0
    assert b.record().phase == "stopped"


def test_kill_with_calls_in_flight_stores_nothing_and_leaves_no_descendant(tmp_path, bridges, monkeypatch):
    """C52 (`--kill` leg), C45 (c own kill) on a real bridge: the helper kills the bridge and the fake claude tree; the
    persona files are as they were before the kill."""
    from brain.bridge import kill

    monkeypatch.setattr(kill, "service_installed", lambda persona: False)
    b = Bridge(tmp_path, calls=(CALLS["compaction"], CALLS["pass2"], CALLS["soul"]), provider="claude-cli",
               prepare=_prepare)
    bridges.append(b)
    pids = _wait_calls(b, 3)
    before = _snapshot(b.pd)
    r = kill.kill_bridge(b.pd, timeout=10)
    assert r.outcome == "killed" and not r.survivors
    assert _all_gone([b.proc.pid, *pids])
    b.proc.wait(10)
    assert _snapshot(b.pd) == before
    assert kill.read_intent(b.pd)["state"] == "committed"
