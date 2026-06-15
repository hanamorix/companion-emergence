"""POST /persona/writes/{rid}/approve and /decline — consent-gated file writes.

Mirrors the existing endpoint-test harness (tests/bridge/test_model_config_endpoint.py):
build the app with build_app(..., auth_token=...), then exercise the routes with and
without the bearer token. Assertions that matter:
  - auth required (401 without bearer)
  - approve commits the pending write (file appears on disk) via commit_write
  - decline resolves via decline_write (no file written)
  - unknown id → 404
  - already-resolved id → 409
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.files import pending


def _make_client(tmp_path: Path, *, auth_token: str | None = None):
    """Minimal persona + TestClient. Returns (client, persona_dir, token)."""
    persona_dir = tmp_path / "personas" / "test"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "fake", "model": "sonnet"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    return TestClient(app), persona_dir, auth_token


def test_approve_requires_auth_and_commits(tmp_path: Path):
    client, persona_dir, token = _make_client(tmp_path, auth_token="secret")
    # Target must be OUTSIDE the persona substrate (the write guard denies
    # any write under persona_dir) and outside the deny-listed system roots.
    target = tmp_path / "workspace" / "a.md"
    rid = pending.create(
        persona_dir, op="create", resolved_path=str(target), content="hi", now=datetime.now(UTC)
    )
    with client:
        # no auth → 401
        assert client.post(f"/persona/writes/{rid}/approve").status_code == 401
        # with auth → commits
        r = client.post(
            f"/persona/writes/{rid}/approve", headers={"Authorization": f"Bearer {token}"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True
    # the write actually landed and the record is resolved
    assert target.read_text(encoding="utf-8") == "hi"
    assert pending.get(persona_dir, rid)["status"] == "committed"


def test_decline_resolves_without_writing(tmp_path: Path):
    client, persona_dir, token = _make_client(tmp_path, auth_token="secret")
    target = tmp_path / "workspace" / "b.md"
    rid = pending.create(
        persona_dir, op="create", resolved_path=str(target), content="hi", now=datetime.now(UTC)
    )
    with client:
        # no auth → 401
        assert client.post(f"/persona/writes/{rid}/decline").status_code == 401
        r = client.post(
            f"/persona/writes/{rid}/decline", headers={"Authorization": f"Bearer {token}"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True
    assert not target.exists()
    assert pending.get(persona_dir, rid)["status"] == "declined"


def test_unknown_id_404(tmp_path: Path):
    client, _persona_dir, token = _make_client(tmp_path, auth_token="secret")
    with client:
        r = client.post(
            "/persona/writes/nope/decline", headers={"Authorization": f"Bearer {token}"}
        )
        assert r.status_code == 404, r.text


def test_double_resolve_409(tmp_path: Path):
    client, persona_dir, token = _make_client(tmp_path, auth_token="secret")
    target = tmp_path / "workspace" / "c.md"
    rid = pending.create(
        persona_dir, op="create", resolved_path=str(target), content="hi", now=datetime.now(UTC)
    )
    headers = {"Authorization": f"Bearer {token}"}
    with client:
        assert client.post(f"/persona/writes/{rid}/approve", headers=headers).status_code == 200
        # second approve on an already-committed write → 409
        assert client.post(f"/persona/writes/{rid}/approve", headers=headers).status_code == 409
        # decline on the same already-resolved write → 409
        assert client.post(f"/persona/writes/{rid}/decline", headers=headers).status_code == 409
