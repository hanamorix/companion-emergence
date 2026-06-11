"""POST /persona/config/pronouns — pronoun persist endpoint tests.

Three contract tests (mirroring test_model_config_endpoint.py):
  1. Preset key accepted + persisted.
  2. Full custom PronounSet dict accepted.
  3. Garbage inputs rejected with 422.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(tmp_path: Path, *, auth_token: str | None = None):
    """Minimal persona + TestClient. Returns (client, persona_dir, headers)."""
    persona_dir = tmp_path / "personas" / "test"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "fake", "model": "sonnet"})
    )
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    return TestClient(app), persona_dir, headers


def test_set_pronouns_preset_persists(tmp_path: Path):
    """A valid preset key is persisted to persona_config.json and resolved correctly."""
    client, persona_dir, _ = _make_client(tmp_path)
    with client:
        r = client.post("/persona/config/pronouns", json={"preset": "he/him"})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    from brain.persona_config import PersonaConfig
    from brain.pronouns import PRESETS, resolve

    cfg = PersonaConfig.load(persona_dir / "persona_config.json")
    assert resolve(cfg.user_pronouns) == PRESETS["he/him"]


def test_set_pronouns_full_set_accepted(tmp_path: Path):
    """A complete custom PronounSet dict is accepted and returns 200."""
    client, persona_dir, _ = _make_client(tmp_path)
    custom = {
        "subject": "xe",
        "object": "xem",
        "possessive": "xyr",
        "possessive_standalone": "xyrs",
        "reflexive": "xemself",
        "plural_verbs": False,
    }
    with client:
        r = client.post("/persona/config/pronouns", json={"set": custom})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    from brain.persona_config import PersonaConfig
    from brain.pronouns import resolve

    cfg = PersonaConfig.load(persona_dir / "persona_config.json")
    resolved = resolve(cfg.user_pronouns)
    assert resolved.subject == "xe"
    assert resolved.object == "xem"


def test_set_pronouns_rejects_garbage(tmp_path: Path):
    """Unknown preset, partial set, and empty body all return 422."""
    client, _, _ = _make_client(tmp_path)
    with client:
        # Unknown preset key
        assert client.post("/persona/config/pronouns", json={"preset": "zir"}).status_code == 422
        # Partial custom set (missing required fields)
        assert client.post(
            "/persona/config/pronouns", json={"set": {"subject": "xe"}}
        ).status_code == 422
        # Empty body — neither preset nor set supplied
        assert client.post("/persona/config/pronouns", json={}).status_code == 422
