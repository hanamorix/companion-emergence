"""#473 (code review 3a, NB4): a deferred close is reported as deferred and the session stays registered with its kept buffer.

`/sessions/close` returned `closed: true` and evicted the session although the deferred pass had kept the buffer for later (before
#473 a refused close was a 502 and the session stayed). Nothing read `IngestReport.deferred`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.ingest.types import IngestReport


def _client(persona_dir: Path) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


def test_a_deferred_close_is_reported_as_deferred_and_the_session_stays(persona_dir: Path, monkeypatch):
    calls = 0

    def fake_close(*args, **kwargs):
        nonlocal calls
        calls += 1
        return IngestReport(session_id=args[1], deferred=(calls == 1))

    monkeypatch.setattr("brain.bridge.server._close_session_blocking", fake_close)
    with _client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        first = c.post("/sessions/close", json={"session_id": sid})
        assert first.status_code == 200  # nothing failed: "not now"
        body = first.json()
        assert body["closed"] is False
        assert body["deferred"] is True
        assert body["errors"] == 0
        assert c.get(f"/state/{sid}").status_code == 200  # still registered, like the kept buffer
        second = c.post("/sessions/close", json={"session_id": sid})
        assert second.json()["closed"] is True and second.json()["deferred"] is False
        assert c.get(f"/state/{sid}").status_code == 404


def test_a_deferred_close_publishes_a_deferred_event_not_a_closed_one(persona_dir: Path, monkeypatch):
    from brain.bridge import events

    seen: list[str] = []
    real = events.publish

    def spy(kind, **kw):
        seen.append(kind)
        return real(kind, **kw)

    monkeypatch.setattr(events, "publish", spy)
    monkeypatch.setattr(
        "brain.bridge.server._close_session_blocking", lambda *a, **k: IngestReport(session_id=a[1], deferred=True)
    )
    with _client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        c.post("/sessions/close", json={"session_id": sid})
    assert "session_closed" not in seen
    assert "session_close_deferred" in seen


def test_the_snapshot_route_reports_deferred_and_busy(persona_dir: Path, monkeypatch):
    reports = iter([IngestReport(session_id="x", deferred=True), IngestReport(session_id="x", busy=True), IngestReport(session_id="x")])
    monkeypatch.setattr("brain.bridge.server._snapshot_session_blocking", lambda *a, **k: next(reports))
    with _client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        bodies = [c.post("/sessions/snapshot", json={"session_id": sid}).json() for _ in range(3)]
    assert [(b["deferred"], b["busy"]) for b in bodies] == [(True, False), (False, True), (False, False)]
