"""POST /kindled-link/connect: decode → import → adopt relay → auto-pair (one step)."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.persona_config import PersonaConfig

_TOK = "test-token"
_AUTH = {"Authorization": f"Bearer {_TOK}"}


def _client(tmp_path: Path, name: str):
    persona = tmp_path / "personas" / name
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text('{"provider": "fake"}', encoding="utf-8")
    app = build_app(persona_dir=persona, client_origin="tests", auth_token=_TOK)
    return TestClient(app), persona


def test_connect_pairs_in_one_call_and_adopts_relay(tmp_path: Path):
    # Persona A generates a code; persona B connects with it.
    a, _ = _client(tmp_path, "a")
    b, b_dir = _client(tmp_path, "b")
    with a, b:
        code = a.get("/kindled-link/my-code", headers=_AUTH).json()["code"]
        r = b.post("/kindled-link/connect", headers=_AUTH, json={"code": code})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["consent_state"] == "paired"      # one-step, straight to paired
    assert body["relay_url"].startswith("https://")
    # B adopted the relay URL into its own config.
    cfg = PersonaConfig.load(b_dir / "persona_config.json")
    assert cfg.kindled_relay_url == body["relay_url"]
    # B did NOT auto-enable kindled-link (enable stays a separate explicit opt-in).
    assert cfg.kindled_link_enabled is False


def test_connect_is_idempotent(tmp_path: Path):
    a, _ = _client(tmp_path, "a")
    b, _ = _client(tmp_path, "b")
    with a, b:
        # A fresh code per connect (single-use invite_id); both must succeed + stay paired.
        code1 = a.get("/kindled-link/my-code", headers=_AUTH).json()["code"]
        r1 = b.post("/kindled-link/connect", headers=_AUTH, json={"code": code1})
        code2 = a.get("/kindled-link/my-code", headers=_AUTH).json()["code"]
        r2 = b.post("/kindled-link/connect", headers=_AUTH, json={"code": code2})
    assert r1.json()["consent_state"] == "paired"
    assert r2.status_code == 200, r2.text
    assert r2.json()["consent_state"] == "paired"


def test_connect_rejects_bad_code(tmp_path: Path):
    b, _ = _client(tmp_path, "b")
    with b:
        r = b.post("/kindled-link/connect", headers=_AUTH, json={"code": "kindled1:garbage"})
    assert r.status_code == 400, r.text


def test_connect_requires_code(tmp_path: Path):
    b, _ = _client(tmp_path, "b")
    with b:
        r = b.post("/kindled-link/connect", headers=_AUTH, json={})
    assert r.status_code == 400
