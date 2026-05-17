"""Tests for GET /images — past-image gallery listing."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(persona_dir: Path, auth_token: str | None = None) -> TestClient:
    app = build_app(persona_dir=persona_dir, auth_token=auth_token)
    return TestClient(app)


def _patch_fake_provider(monkeypatch):
    """Replace get_provider so bridge lifespan doesn't need a real LLM."""
    import brain.bridge.server as srv
    from brain.bridge.chat import ChatResponse

    class _FakeProvider:
        def chat(self, *args, **kwargs):
            return ChatResponse(
                session_id="fake",
                reply="ok",
                turn=1,
                tool_invocations=[],
                duration_ms=1,
                metadata={"persistence_ok": True},
            )

        def extract_memories(self, *args, **kwargs):
            return []

    monkeypatch.setattr(srv, "get_provider", lambda _name=None: _FakeProvider())


def _auth_headers(token: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _write_buffer(persona_dir: Path, session_id: str, turns: list[dict]) -> Path:
    buf_dir = persona_dir / "active_conversations"
    buf_dir.mkdir(parents=True, exist_ok=True)
    buf_path = buf_dir / f"{session_id}.jsonl"
    with open(buf_path, "w", encoding="utf-8") as fh:
        for turn in turns:
            fh.write(json.dumps(turn) + "\n")
    return buf_path


def _write_real_image(persona_dir: Path, ext: str) -> tuple[str, str]:
    data = {
        "png": b"\x89PNG\r\n\x1a\n" + uuid.uuid4().bytes,
        "jpg": b"\xff\xd8\xff" + uuid.uuid4().bytes,
        "webp": b"RIFF\x10\x00\x00\x00WEBP" + uuid.uuid4().bytes,
        "gif": b"GIF89a" + uuid.uuid4().bytes,
    }[ext]
    sha = hashlib.sha256(data).hexdigest()
    images_dir = persona_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / f"{sha}.{ext}").write_bytes(data)
    return sha, ext


class TestListImages:
    def test_empty_persona_returns_empty_list(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "empty-persona"
        persona_dir.mkdir()
        with _make_client(persona_dir) as c:
            r = c.get("/images")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_images_from_buffer(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha1, ext1 = _write_real_image(persona_dir, "png")
        sha2, ext2 = _write_real_image(persona_dir, "jpg")
        _write_buffer(
            persona_dir,
            "session-1",
            [
                {"ts": "2026-05-01T10:00:00Z", "image_shas": [sha1]},
                {"ts": "2026-05-01T11:00:00Z", "image_shas": [sha2]},
            ],
        )
        with _make_client(persona_dir) as c:
            r = c.get("/images")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["sha"] == sha2
        assert data[1]["sha"] == sha1

    def test_dedupes_same_sha_across_sessions(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha, _ = _write_real_image(persona_dir, "png")
        _write_buffer(
            persona_dir,
            "session-1",
            [
                {"ts": "2026-05-02T10:00:00Z", "image_shas": [sha]},
            ],
        )
        _write_buffer(
            persona_dir,
            "session-2",
            [
                {"ts": "2026-05-01T09:00:00Z", "image_shas": [sha]},
            ],
        )
        with _make_client(persona_dir) as c:
            r = c.get("/images")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["first_seen_ts"] == "2026-05-01T09:00:00Z"

    def test_respects_limit(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        for i in range(10):
            sha, _ = _write_real_image(persona_dir, "png")
            _write_buffer(
                persona_dir,
                f"session-{i}",
                [
                    {"ts": f"2026-05-{i + 1:02d}T10:00:00Z", "image_shas": [sha]},
                ],
            )
        with _make_client(persona_dir) as c:
            r = c.get("/images", params={"limit": 3})
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_before_ts_filters(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha1, _ = _write_real_image(persona_dir, "png")
        sha2, _ = _write_real_image(persona_dir, "jpg")
        _write_buffer(
            persona_dir,
            "session-1",
            [
                {"ts": "2026-05-01T10:00:00Z", "image_shas": [sha1]},
                {"ts": "2026-05-05T10:00:00Z", "image_shas": [sha2]},
            ],
        )
        with _make_client(persona_dir) as c:
            r = c.get("/images", params={"before_ts": "2026-05-03T00:00:00Z"})
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["sha"] == sha1

    def test_skips_missing_image_files(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha_present, _ = _write_real_image(persona_dir, "png")
        _write_buffer(
            persona_dir,
            "session-1",
            [
                {"ts": "2026-05-01T10:00:00Z", "image_shas": [sha_present]},
                {"ts": "2026-05-01T11:00:00Z", "image_shas": ["a" * 64]},
            ],
        )
        with _make_client(persona_dir) as c:
            r = c.get("/images")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["sha"] == sha_present

    def test_skips_corrupt_buffer_lines(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha, _ = _write_real_image(persona_dir, "png")
        buf_dir = persona_dir / "active_conversations"
        buf_dir.mkdir(parents=True)
        with open(buf_dir / "session-1.jsonl", "w") as fh:
            fh.write(json.dumps({"ts": "2026-05-01T10:00:00Z", "image_shas": [sha]}) + "\n")
            fh.write("not valid json\n")
        with _make_client(persona_dir) as c:
            r = c.get("/images")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_requires_auth_when_token_set(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha, _ = _write_real_image(persona_dir, "png")
        _write_buffer(
            persona_dir,
            "session-1",
            [
                {"ts": "2026-05-01T10:00:00Z", "image_shas": [sha]},
            ],
        )
        with _make_client(persona_dir, auth_token="secret") as c:
            r = c.get("/images")
        assert r.status_code == 401

    def test_accepts_valid_auth(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha, _ = _write_real_image(persona_dir, "png")
        _write_buffer(
            persona_dir,
            "session-1",
            [
                {"ts": "2026-05-01T10:00:00Z", "image_shas": [sha]},
            ],
        )
        with _make_client(persona_dir, auth_token="secret") as c:
            r = c.get("/images", headers=_auth_headers("secret"))
        assert r.status_code == 200


class TestServeImage:
    def test_serves_image_bytes(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        sha, _ = _write_real_image(persona_dir, "png")
        with _make_client(persona_dir) as c:
            r = c.get(f"/images/{sha}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_unknown_sha_returns_404(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        with _make_client(persona_dir) as c:
            r = c.get("/images/" + "a" * 64)
        assert r.status_code == 404

    def test_malformed_sha_returns_404(self, tmp_path: Path, monkeypatch):
        _patch_fake_provider(monkeypatch)
        persona_dir = tmp_path / "test-persona"
        persona_dir.mkdir()
        with _make_client(persona_dir) as c:
            r = c.get("/images/not-a-sha")
        assert r.status_code == 404
