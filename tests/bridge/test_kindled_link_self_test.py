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
