"""Tests for POST /supervisor/shutdown — the manual restart endpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(persona_dir: Path) -> TestClient:
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return TestClient(app)


class FakeShutdownController:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def request(self, reason: str) -> bool:
        self.reasons.append(reason)
        return True


def test_post_supervisor_shutdown_returns_202_immediately(persona_dir: Path):
    """Endpoint returns 202 Accepted without blocking on the 30s drain."""
    controller = FakeShutdownController()
    app = build_app(
        persona_dir=persona_dir,
        client_origin="tests",
        shutdown_controller=controller,
    )
    with TestClient(app) as c:
        r = c.post("/supervisor/shutdown")
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "shutting_down"
        assert body["drain_seconds"] == 30


def test_post_supervisor_shutdown_requires_auth(persona_dir: Path):
    """Missing or wrong bearer token → 401."""
    controller = FakeShutdownController()
    app = build_app(
        persona_dir=persona_dir,
        client_origin="tests",
        auth_token="secret-token-for-test",
        shutdown_controller=controller,
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

        # Correct token → 202
        r = c.post(
            "/supervisor/shutdown",
            headers={"Authorization": "Bearer secret-token-for-test"},
        )
        assert r.status_code == 202


def test_post_supervisor_shutdown_uses_controller_not_os_kill(persona_dir: Path):
    controller = FakeShutdownController()
    app = build_app(
        persona_dir=persona_dir,
        client_origin="tests",
        shutdown_controller=controller,
    )
    with patch("brain.bridge.server.os.kill", side_effect=AssertionError("must not signal")):
        with TestClient(app) as c:
            r = c.post("/supervisor/shutdown")

    assert r.status_code == 202
    assert controller.reasons == ["manual_restart"]


def test_post_supervisor_shutdown_reports_unavailable_without_controller(persona_dir: Path):
    app = build_app(persona_dir=persona_dir, client_origin="tests", shutdown_controller=None)
    with TestClient(app) as c:
        r = c.post("/supervisor/shutdown")

    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "shutdown_controller_unavailable"


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
