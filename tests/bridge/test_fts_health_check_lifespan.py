"""INC-3 (spec §6c): the bridge lifespan calls `run_fts_health_check_once`
once, before the first `MemoryStore` open, and fail-soft — a bug in this
brand-new check must never be the reason a bridge fails to start
(code-red-team pass 1, M1)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge import server


def test_fts_health_check_called_before_first_store_open(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_check(db_path):
        calls.append("check")

    real_memory_store = server.MemoryStore

    def tracking_store(*args, **kwargs):
        calls.append("store_open")
        return real_memory_store(*args, **kwargs)

    monkeypatch.setattr(server, "run_fts_health_check_once", fake_check)
    monkeypatch.setattr(server, "MemoryStore", tracking_store)

    app = server.build_app(persona_dir=persona_dir, client_origin="tests")
    with TestClient(app):
        pass

    assert "check" in calls
    assert calls.index("check") < calls.index("store_open"), (
        "the health check must run before the first MemoryStore open"
    )


def test_fts_health_check_failure_does_not_block_bridge_startup(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    def boom(db_path):
        raise RuntimeError("simulated db_health bug")

    monkeypatch.setattr(server, "run_fts_health_check_once", boom)

    app = server.build_app(persona_dir=persona_dir, client_origin="tests")
    with caplog.at_level(logging.WARNING, logger="brain.bridge.server"):
        with TestClient(app) as c:
            # Fail-first: before this fix, a raising health check was not
            # wrapped in try/except and would abort lifespan startup — the
            # app would never come up and this request would never get a
            # response.
            assert c.get("/health").status_code == 200
    assert any("FTS health check skipped" in r.getMessage() for r in caplog.records)
