"""POST /kindled-link/self-test wires the bridge to run_self_test (monkeypatched)."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app

_TOK = "test-token"
_AUTH = {"Authorization": f"Bearer {_TOK}"}


def _client(tmp_path: Path):
    persona = tmp_path / "personas" / "nell"
    persona.mkdir(parents=True)
    # kindled_link_enabled=True: self-test is gated behind explicit opt-in (the
    # relay-gate fix) — this suite exercises the self-test mechanics, not the gate.
    (persona / "persona_config.json").write_text(
        '{"provider": "fake", "kindled_link_enabled": true}', encoding="utf-8"
    )
    return TestClient(build_app(persona_dir=persona, client_origin="tests", auth_token=_TOK)), persona


def test_self_test_endpoint_returns_stage_report(tmp_path: Path, monkeypatch):
    fake = {"ok": True, "stages": [{"name": "relay_reachable", "ok": True, "detail": ""}],
            "relay_url": "https://relay.test"}
    monkeypatch.setattr("brain.kindled_link.self_test.run_self_test", lambda *a, **k: fake)
    client, _ = _client(tmp_path)
    with client:
        r = client.post("/kindled-link/self-test", headers=_AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert r.json()["stages"][0]["name"] == "relay_reachable"


def test_self_test_endpoint_requires_auth(tmp_path: Path):
    client, _ = _client(tmp_path)
    with client:
        r = client.post("/kindled-link/self-test")
    assert r.status_code in (401, 403)


def test_self_test_is_a_counted_stop_task_while_it_runs(tmp_path: Path, monkeypatch):
    """Review of #458 (finding 4): a synchronous route is a counted bridge task for as long as its thread runs, so a
    stop waits for it. Its relay round-trips (30 s timeout each) outlast uvicorn's 30 s graceful window, after which
    the request task is cancelled but the thread keeps writing the kindled-link store."""
    from brain.bridge import shutdown_progress

    seen: list[list[str]] = []

    def _fake(*_a, **_k):
        seen.append(shutdown_progress.registry.snapshot())
        return {"ok": True, "stages": [], "relay_url": "https://relay.test"}

    monkeypatch.setattr("brain.kindled_link.self_test.run_self_test", _fake)
    client, _ = _client(tmp_path)
    with client:
        r = client.post("/kindled-link/self-test", headers=_AUTH)
    assert r.status_code == 200, r.text
    assert any("/kindled-link/self-test" in label for label in seen[0]), seen


def test_a_sync_route_after_the_seal_gets_the_same_bridge_stopping_body(tmp_path: Path, monkeypatch):
    """A synchronous route whose worker registers after the seal is refused with the admission middleware's body
    (top-level ``code``), which the app's restart flow reads."""
    from brain.bridge import shutdown_progress

    ran: list[int] = []
    monkeypatch.setattr("brain.kindled_link.self_test.run_self_test", lambda *_a, **_k: ran.append(1) or {})
    client, _ = _client(tmp_path)
    with client:
        assert shutdown_progress.registry.seal() is True
        r = client.post("/kindled-link/self-test", headers=_AUTH)
    assert r.status_code == 503 and r.json()["code"] == "bridge_stopping", r.text
    assert ran == []
