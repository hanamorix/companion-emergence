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
                 status="hold")  # live status string (stage-6 fix: was 'held')
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


def test_create_invite_returns_packet_and_phrase(tmp_path):
    app, _ = _app(tmp_path)
    r = TestClient(app).post("/kindled-link/invite", headers=_AUTH,
                             json={"relay_url": "https://r"})
    assert r.status_code == 200
    body = r.json()
    assert "invite" in body and "fingerprint" in body and "fingerprint_phrase" in body


def test_accept_invite_creates_peer(tmp_path):
    # build a real invite from a SECOND throwaway identity (the remote peer)
    from brain.kindled_link.identity import KindledIdentity
    from brain.kindled_link.pairing import create_invite
    remote = KindledIdentity.load_or_create(tmp_path / "remote")
    invite = create_invite(remote, relay_url="https://r")
    app, _ = _app(tmp_path)
    r = TestClient(app).post("/kindled-link/invite/accept", headers=_AUTH,
                             json={"invite": invite})
    assert r.status_code == 200
    assert "peer_id" in r.json() and "fingerprint_phrase" in r.json()


def test_accept_missing_invite_is_400(tmp_path):
    app, _ = _app(tmp_path)
    r = TestClient(app).post("/kindled-link/invite/accept", headers=_AUTH, json={})
    assert r.status_code == 400


def _seed_peer(persona, *, consent_state):
    s = _seed_store(persona)
    s.upsert_peer(peer_id="kid_a", identity_pub_hex="aa", fingerprint="kid_a",
                  consent_state=consent_state, relay_url="https://r", now=NOW)


def test_consent_pause_paired_peer(tmp_path):
    app, persona = _app(tmp_path)
    _seed_peer(persona, consent_state="paired")
    r = TestClient(app).post("/kindled-link/peers/kid_a/consent", headers=_AUTH,
                             json={"action": "pause"})
    assert r.status_code == 200 and r.json()["consent_state"] == "paused"


def test_consent_illegal_transition_is_400(tmp_path):
    # blocked is terminal — resume must be rejected, not silently applied
    app, persona = _app(tmp_path)
    _seed_peer(persona, consent_state="blocked")
    r = TestClient(app).post("/kindled-link/peers/kid_a/consent", headers=_AUTH,
                             json={"action": "resume"})
    assert r.status_code == 400


def test_consent_unknown_action_is_400(tmp_path):
    app, persona = _app(tmp_path)
    _seed_peer(persona, consent_state="paired")
    r = TestClient(app).post("/kindled-link/peers/kid_a/consent", headers=_AUTH,
                             json={"action": "frobnicate"})
    assert r.status_code == 400
