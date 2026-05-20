"""Tests for POST /supervisor/shutdown — the manual restart endpoint."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain.bridge.server import build_app


@pytest.fixture(autouse=True)
def _patch_os_kill():
    """Block `brain.bridge.server.os.kill` for every test in this module.

    The POST /supervisor/shutdown handler schedules an async task that, 100ms
    after the response, calls `os.kill(os.getpid(), SIGTERM)`. Per-test
    `with patch(...)` scopes exit when the test's `with TestClient(...)` block
    exits — but the asyncio.create_task() fire-and-forget can still complete
    *after* the patch is torn down, calling the real `os.kill` on the running
    pytest process. On POSIX that signal is caught and ignored by pytest's
    default handlers, so the next test runs fine. On Windows
    `os.kill(pid, SIGTERM)` maps to TerminateProcess — pytest dies mid-test,
    exit code 1, no FAILED traceback. Patching at module scope outlives the
    deferred task and survives the platform difference.
    """
    with patch("brain.bridge.server.os.kill"):
        yield


def _make_client(persona_dir: Path) -> TestClient:
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return TestClient(app)


def test_post_supervisor_shutdown_returns_202_immediately(persona_dir: Path):
    """Endpoint returns 202 Accepted without blocking on the 30s drain."""
    with _make_client(persona_dir) as c:
        r = c.post("/supervisor/shutdown")
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "shutting_down"
        assert body["drain_seconds"] == 30


def test_post_supervisor_shutdown_requires_auth(persona_dir: Path):
    """Missing or wrong bearer token → 401."""
    app = build_app(
        persona_dir=persona_dir,
        client_origin="tests",
        auth_token="secret-token-for-test",
    )
    with TestClient(app) as c:
        # No Authorization header → 401
        r = c.post("/supervisor/shutdown")
        assert r.status_code == 401

        # Wrong token → 401
        r = c.post(
            "/supervisor/shutdown",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert r.status_code == 401

        # Correct token → 202 (autouse fixture has os.kill patched)
        r = c.post(
            "/supervisor/shutdown",
            headers={"Authorization": "Bearer secret-token-for-test"},
        )
        assert r.status_code == 202


def test_post_supervisor_shutdown_schedules_sigterm(persona_dir: Path):
    """The endpoint schedules a SIGTERM to self after returning 202."""
    import time

    with _make_client(persona_dir) as c:
        with patch("brain.bridge.server.os.kill") as mock_kill:
            r = c.post("/supervisor/shutdown")
            assert r.status_code == 202
            # 100ms delay before SIGTERM — wait a beat for the deferred task.
            time.sleep(0.3)
        assert mock_kill.call_count >= 1
        call_args = mock_kill.call_args_list[0]
        assert call_args.args[0] == os.getpid()
        assert call_args.args[1] == signal.SIGTERM


def test_bridge_json_includes_pid_field(persona_dir: Path):
    """bridge.json after state_file.write contains a positive int pid matching os.getpid()."""
    from datetime import UTC, datetime

    from brain.bridge import state_file
    from brain.bridge.state_file import BridgeState

    state = BridgeState(
        persona=persona_dir.name,
        pid=os.getpid(),
        port=12345,
        started_at=datetime.now(UTC).isoformat(),
        stopped_at=None,
        shutdown_clean=False,
        client_origin="tests",
        auth_token="t",
    )
    state_file.write(persona_dir, state)

    bridge_json = persona_dir / "bridge.json"
    assert bridge_json.exists()
    data = json.loads(bridge_json.read_text(encoding="utf-8"))
    assert isinstance(data["pid"], int)
    assert data["pid"] > 0
    assert data["pid"] == os.getpid()


def test_bridge_json_pid_field_atomic(persona_dir: Path):
    """save_with_backup writes via tmpfile+rename — every read sees a parseable file with pid."""
    from datetime import UTC, datetime

    from brain.bridge import state_file
    from brain.bridge.state_file import BridgeState

    state = BridgeState(
        persona=persona_dir.name,
        pid=os.getpid(),
        port=12345,
        started_at=datetime.now(UTC).isoformat(),
        stopped_at=None,
        shutdown_clean=False,
        client_origin="tests",
        auth_token="t",
    )

    for _ in range(10):
        state_file.write(persona_dir, state)
        data = json.loads((persona_dir / "bridge.json").read_text(encoding="utf-8"))
        assert "pid" in data
        assert data["pid"] == os.getpid()
