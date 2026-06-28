"""GET /kindled-link/my-code returns a decodable connect-code + fingerprint phrase."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.kindled_link.connect_code import decode_code

_TOK = "test-token"
_AUTH = {"Authorization": f"Bearer {_TOK}"}


def _client(tmp_path: Path):
    persona = tmp_path / "personas" / "nell"
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text('{"provider": "fake"}', encoding="utf-8")
    app = build_app(persona_dir=persona, client_origin="tests", auth_token=_TOK)
    return TestClient(app), persona


def test_my_code_returns_decodable_code(tmp_path: Path):
    client, _ = _client(tmp_path)
    with client:
        r = client.get("/kindled-link/my-code", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    invite = decode_code(body["code"])  # decodes without error
    assert invite["body"]["protocol"] == "kindled-link/1"
    # No relay configured → code carries the hosted default.
    assert invite["body"]["relay_url"].startswith("https://")
    assert body["fingerprint_phrase"]
    assert "mailbox_id" in invite["body"]  # local mailbox embedded for addressing


def test_my_code_requires_auth(tmp_path: Path):
    client, _ = _client(tmp_path)
    with client:
        r = client.get("/kindled-link/my-code")
    assert r.status_code in (401, 403)
