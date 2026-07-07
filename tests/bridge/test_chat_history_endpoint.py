"""GET /chat/history reads from active_conversations/<sid>.jsonl.

Backs the v0.0.15-alpha.2 chat-reliability Phase 3A endpoint that lets the
renderer hydrate its message list from disk on reopen. The buffer is the
canonical conversation log — one JSONL line per turn, ``{session_id,
speaker, text, ts}`` — and this endpoint exposes it through the bridge so
the frontend never has to touch the persona dir directly.

Conftest fixtures used:
  - ``persona_dir`` — tmp ``$KINDLED_HOME``-shaped dir with empty
    ``active_conversations/`` and a minimal ``persona_config.json``.
  - ``_reset_session_registry`` — autouse, clears the in-memory registry
    between tests.

Each test stands up its own ``TestClient`` via ``_make_client`` (mirrors
``tests/bridge/test_endpoints.py``) so the test app's lifespan owns its
own provider stub.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(persona_dir: Path, *, auth_token: str | None = None) -> TestClient:
    """Build a TestClient pinned to ``persona_dir``.

    ``auth_token`` defaults to None so the endpoint tests don't need to
    pass an Authorization header. The auth-rejection test flips it on.
    """
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=auth_token)
    return TestClient(app)


def _seed_buffer(persona_dir: Path, session_id: str, turns: list[dict]) -> None:
    """Write JSONL turns to ``<persona>/active_conversations/<sid>.jsonl``.

    Turns use the on-disk schema (``speaker``, ``text``, ``ts``) — the
    endpoint's job is to translate that to the renderer-friendly
    ``role``/``content`` shape.
    """
    buffers = persona_dir / "active_conversations"
    buffers.mkdir(parents=True, exist_ok=True)
    path = buffers / f"{session_id}.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in turns), encoding="utf-8")


def test_history_returns_empty_array_for_missing_session(persona_dir: Path) -> None:
    with _make_client(persona_dir) as c:
        r = c.get("/chat/history", params={"session_id": "s_missing"})
        assert r.status_code == 200
        assert r.json() == {"messages": [], "next_before_turn": None}


def test_history_returns_buffered_turns_in_order(persona_dir: Path) -> None:
    _seed_buffer(
        persona_dir,
        "s_a",
        [
            {"session_id": "s_a", "speaker": "user", "text": "hi", "ts": "2026-05-20T10:00:00Z"},
            {
                "session_id": "s_a",
                "speaker": "assistant",
                "text": "hello",
                "ts": "2026-05-20T10:00:05Z",
            },
        ],
    )
    with _make_client(persona_dir) as c:
        r = c.get("/chat/history", params={"session_id": "s_a"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["messages"]) == 2
    assert body["messages"][0]["content"] == "hi"
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["turn"] == 1
    assert body["messages"][1]["content"] == "hello"
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][1]["turn"] == 2
    assert body["next_before_turn"] is None


def test_history_respects_limit(persona_dir: Path) -> None:
    _seed_buffer(
        persona_dir,
        "s_b",
        [
            {
                "session_id": "s_b",
                "speaker": "user",
                "text": f"m{i}",
                "ts": "2026-05-20T10:00:00Z",
            }
            for i in range(10)
        ],
    )
    with _make_client(persona_dir) as c:
        r = c.get("/chat/history", params={"session_id": "s_b", "limit": 3})
    assert r.status_code == 200
    body = r.json()
    # limit returns the *tail* (most recent turns) — that's what the renderer
    # wants on initial hydration, with older turns paged in via before_turn.
    assert len(body["messages"]) == 3
    assert [m["content"] for m in body["messages"]] == ["m7", "m8", "m9"]
    assert [m["turn"] for m in body["messages"]] == [8, 9, 10]
    # next_before_turn lets the client request the page above ↑.
    assert body["next_before_turn"] == 8


def test_history_pagination_via_before_turn(persona_dir: Path) -> None:
    _seed_buffer(
        persona_dir,
        "s_c",
        [
            {
                "session_id": "s_c",
                "speaker": "user",
                "text": f"m{i}",
                "ts": "2026-05-20T10:00:00Z",
            }
            for i in range(10)
        ],
    )
    with _make_client(persona_dir) as c:
        r = c.get(
            "/chat/history",
            params={"session_id": "s_c", "limit": 3, "before_turn": 5},
        )
    assert r.status_code == 200
    body = r.json()
    # before_turn=5 → turns 2, 3, 4 (tail of [1..4] capped at limit=3).
    assert [m["turn"] for m in body["messages"]] == [2, 3, 4]
    assert all(m["turn"] < 5 for m in body["messages"])
    assert body["next_before_turn"] == 2


def test_history_rejects_unauthenticated_request(persona_dir: Path) -> None:
    with _make_client(persona_dir, auth_token="secret-token") as c:
        r = c.get("/chat/history", params={"session_id": "s_a"})
        assert r.status_code == 401


def test_history_skips_corrupt_jsonl_lines(persona_dir: Path) -> None:
    path = persona_dir / "active_conversations" / "s_d.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"session_id": "s_d", "speaker": "user", "text": "ok1", "ts": "x"}\n'
        "{this is not valid json}\n"
        '{"session_id": "s_d", "speaker": "user", "text": "ok2", "ts": "x"}\n',
        encoding="utf-8",
    )
    with _make_client(persona_dir) as c:
        r = c.get("/chat/history", params={"session_id": "s_d"})
    assert r.status_code == 200
    body = r.json()
    # Corrupt line is skipped; turn numbers track surviving lines 1-based
    # against the on-disk order so the client can paginate stably.
    assert [m["content"] for m in body["messages"]] == ["ok1", "ok2"]


def test_history_rejects_invalid_session_id(persona_dir: Path) -> None:
    """Path-traversal characters fail the session_id grammar check."""
    with _make_client(persona_dir) as c:
        r = c.get("/chat/history", params={"session_id": "../../etc/passwd"})
        assert r.status_code == 400


def test_history_early_break_closes_reader_generator(
    persona_dir: Path, monkeypatch
) -> None:
    """Windows-deletability hazard: the endpoint breaks out of the buffer
    iteration when ``before_turn`` is hit. An abandoned generator keeps the
    file handle open until GC — on Windows that blocks close_session's
    buffer unlink (WinError 32). Pin that the endpoint closes the reader
    deterministically."""
    import brain.bridge.server as server_mod

    _seed_buffer(
        persona_dir,
        "s_close",
        [
            {"session_id": "s_close", "speaker": "user", "text": f"t{i}", "ts": "x"}
            for i in range(10)
        ],
    )

    closed = {"v": False}
    real_iter = server_mod.iter_jsonl_skipping_corrupt

    def tracking_iter(path):
        gen = real_iter(path)

        class _Wrap:
            def __iter__(self):
                return self

            def __next__(self):
                return next(gen)

            def close(self):
                closed["v"] = True
                gen.close()

        return _Wrap()

    monkeypatch.setattr(server_mod, "iter_jsonl_skipping_corrupt", tracking_iter)

    with _make_client(persona_dir) as c:
        r = c.get("/chat/history", params={"session_id": "s_close", "before_turn": 3})
    assert r.status_code == 200
    assert closed["v"], "endpoint abandoned the buffer reader without closing it"
