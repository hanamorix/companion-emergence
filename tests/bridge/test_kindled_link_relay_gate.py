"""A never-enabled, never-configured install must not dial the default relay.

Gate rule: the DEFAULT-relay fallback (and any outbound dial) fires only when
the user has explicitly opted in — `kindled_link_enabled is True` OR
`kindled_relay_url is not None`. Otherwise /self-test and /connect return a
clear, non-dialling 400 instead of silently wiring + contacting the hosted
relay. See docs/superpowers/specs/2026-06-30-kindled-link-mind-wiring-and-relay-gate.md §C.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.persona_config import PersonaConfig

_TOK = "test-token"
_AUTH = {"Authorization": f"Bearer {_TOK}"}


def _client(tmp_path: Path, *, enabled: bool = False, relay_url: str | None = None,
            name: str = "nell"):
    persona = tmp_path / "personas" / name
    persona.mkdir(parents=True)
    cfg: dict = {"provider": "fake", "kindled_link_enabled": enabled}
    if relay_url is not None:
        cfg["kindled_relay_url"] = relay_url
    (persona / "persona_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return TestClient(build_app(persona_dir=persona, client_origin="tests", auth_token=_TOK)), persona


def test_self_test_disabled_and_unset_returns_400_no_dial(tmp_path: Path, monkeypatch):
    called = {"hit": False}

    def _spy(*a, **k):
        called["hit"] = True
        return {"ok": True, "stages": [], "relay_url": "x"}

    monkeypatch.setattr("brain.kindled_link.self_test.run_self_test", _spy)
    client, _ = _client(tmp_path, enabled=False, relay_url=None)
    with client:
        r = client.post("/kindled-link/self-test", headers=_AUTH)
    assert r.status_code == 400, r.text
    assert r.json()["error"] == "kindled_link_not_enabled"
    assert called["hit"] is False


def test_self_test_enabled_proceeds(tmp_path: Path, monkeypatch):
    fake = {"ok": True, "stages": [{"name": "relay_reachable", "ok": True, "detail": ""}],
            "relay_url": "https://relay.test"}
    monkeypatch.setattr("brain.kindled_link.self_test.run_self_test", lambda *a, **k: fake)
    client, _ = _client(tmp_path, enabled=True, relay_url=None)
    with client:
        r = client.post("/kindled-link/self-test", headers=_AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


def test_self_test_disabled_and_unset_creates_no_kindled_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "brain.kindled_link.self_test.run_self_test",
        lambda *a, **k: {"ok": True, "stages": [], "relay_url": "x"},
    )
    client, persona = _client(tmp_path, enabled=False, relay_url=None)
    with client:
        client.post("/kindled-link/self-test", headers=_AUTH)
    assert not (persona / "kindled_link").exists()


def test_self_test_disabled_but_relay_set_proceeds(tmp_path: Path, monkeypatch):
    # Explicit relay set counts as opt-in even if the toggle itself is off.
    fake = {"ok": True, "stages": [], "relay_url": "https://custom.example.com"}
    monkeypatch.setattr("brain.kindled_link.self_test.run_self_test", lambda *a, **k: fake)
    client, _ = _client(tmp_path, enabled=False, relay_url="https://custom.example.com")
    with client:
        r = client.post("/kindled-link/self-test", headers=_AUTH)
    assert r.status_code == 200, r.text


def test_connect_disabled_and_unset_returns_400_config_not_mutated(tmp_path: Path):
    client, persona = _client(tmp_path, enabled=False, relay_url=None)
    with client:
        r = client.post("/kindled-link/connect", headers=_AUTH, json={"code": "kindled1:garbage"})
    assert r.status_code == 400, r.text
    assert r.json().get("error") == "kindled_link_not_enabled"
    cfg = PersonaConfig.load(persona / "persona_config.json")
    assert cfg.kindled_relay_url is None


def test_connect_disabled_and_unset_creates_no_kindled_dir(tmp_path: Path):
    client, persona = _client(tmp_path, enabled=False, relay_url=None)
    with client:
        client.post("/kindled-link/connect", headers=_AUTH, json={"code": "kindled1:garbage"})
    assert not (persona / "kindled_link").exists()


def test_connect_enabled_proceeds(tmp_path: Path):
    a, _ = _client(tmp_path, enabled=True, relay_url=None, name="a")
    b, _ = _client(tmp_path, enabled=True, relay_url=None, name="b")
    with a, b:
        code = a.get("/kindled-link/my-code", headers=_AUTH).json()["code"]
        r = b.post("/kindled-link/connect", headers=_AUTH, json={"code": code})
    assert r.status_code == 200, r.text
    assert r.json()["consent_state"] == "paired"
