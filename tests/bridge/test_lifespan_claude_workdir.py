"""#122 — the bridge reports the claude working directory at start (C10)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge import provider as provider_mod
from brain.bridge.server import build_app


def test_lifespan_logs_chosen_workdir(persona_dir: Path, tmp_path: Path, monkeypatch, caplog):
    t = tmp_path / "t"
    t.mkdir()
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    with caplog.at_level(logging.INFO, logger="brain.bridge.server"):
        with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
            pass
    msgs = [r.getMessage() for r in caplog.records if "claude working directory" in r.getMessage()]
    assert msgs and str(t) in msgs[0]


def test_lifespan_warns_when_no_neutral_dir(persona_dir: Path, tmp_path: Path, monkeypatch, caplog):
    t = tmp_path / "t"
    t.mkdir()
    (t / "CLAUDE.md").write_text("x")
    monkeypatch.setattr(provider_mod.tempfile, "gettempdir", lambda: str(t))
    kh = tmp_path / "kindled-home"  # the conftest fixture points KINDLED_HOME here
    kh.mkdir(exist_ok=True)
    (kh / ".claude").mkdir(exist_ok=True)  # second candidate also dirty
    # The seam, not sys.platform: faking the whole platform leaks into process-wide caches (#662).
    monkeypatch.setattr(provider_mod, "_is_windows", lambda: True)
    provider_mod._claude_work_dir_reset_for_tests()
    with caplog.at_level(logging.WARNING):
        with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
            pass
    assert any("#252" in r.getMessage() for r in caplog.records)
