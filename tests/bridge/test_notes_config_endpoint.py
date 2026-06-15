"""POST /persona/config/notes — enable/disable autonomous notes.

Mirrors the test_model_config_endpoint harness (build_app + TestClient +
optional bearer auth). On enable, the endpoint resolves the per-OS folder
(platformdirs Documents / '<Persona> Notes'), creates it, and persists it.
The client supplies no path — the system picks the folder.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(tmp_path: Path, *, auth_token: str | None = None):
    """Minimal persona + TestClient. Returns (client, persona_dir)."""
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop", "model": "sonnet"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    return TestClient(app), persona_dir


def test_enable_notes_resolves_and_creates_folder(tmp_path: Path, monkeypatch):
    client, persona_dir = _make_client(tmp_path)
    # point Documents at a temp dir
    import brain.notes.config as ncfg
    monkeypatch.setattr(ncfg, "user_documents_dir", lambda: str(tmp_path / "Documents"))
    with client:
        r = client.post("/persona/config/notes", json={"enabled": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["enabled"] is True
    assert body["folder"].endswith("Notes")
    assert (tmp_path / "Documents" / "nell Notes").exists()  # folder created


def test_notes_requires_auth(tmp_path: Path):
    client, _persona_dir = _make_client(tmp_path, auth_token="secret")
    with client:
        r = client.post("/persona/config/notes", json={"enabled": True})
    assert r.status_code == 401, r.text
