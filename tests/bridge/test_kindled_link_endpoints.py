from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.kindled_link.store import KindledLinkStore, kindled_db_path

NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
_TOK = "secret-token"
_AUTH = {"Authorization": f"Bearer {_TOK}"}


def _app(tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir(parents=True, exist_ok=True)
    return build_app(persona_dir=persona, client_origin="tests", auth_token=_TOK), persona


def _seed_store(persona):
    db = kindled_db_path(persona)
    db.parent.mkdir(parents=True, exist_ok=True)
    return KindledLinkStore(db)


def test_get_peers_returns_seeded_peer(tmp_path):
    app, persona = _app(tmp_path)
    s = _seed_store(persona)
    s.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="kid_a",
                  consent_state="paired", relay_url="https://r", now=NOW)
    r = TestClient(app).get("/kindled-link/peers", headers=_AUTH)
    assert r.status_code == 200
    assert any(p["peer_id"] == "kid_a" for p in r.json()["peers"])


def test_get_holds_never_contains_body_at_http_layer(tmp_path):
    # THE SPINE at the HTTP layer.
    app, persona = _app(tmp_path)
    s = _seed_store(persona)
    s.save_draft(peer_id="kid_a", session_id="s1",
                 payload_json='{"body": "SECRET_USER_DETAIL_SENTINEL"}', now=NOW,
                 status="held")
    r = TestClient(app).get("/kindled-link/holds", headers=_AUTH)
    assert r.status_code == 200 and r.json()["held_count"] == 1
    assert "SECRET_USER_DETAIL_SENTINEL" not in r.text


_GET_ROUTES = [
    ("GET", "/kindled-link/peers"),
    ("GET", "/kindled-link/peers/kid_a/transcript"),
    ("GET", "/kindled-link/holds"),
]


@pytest.mark.parametrize("method,path", _GET_ROUTES)
def test_all_kindled_routes_reject_missing_auth(tmp_path, method, path):
    # B2 / m9: All /kindled-link/ GET routes are auth-gated.
    app, _ = _app(tmp_path)
    client = TestClient(app)
    r = client.request(method, path, json={})
    assert r.status_code in (401, 403), f"{method} {path} not auth-gated"
