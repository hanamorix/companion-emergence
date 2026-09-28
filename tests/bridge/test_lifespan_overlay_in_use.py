"""#302 — a bridge running an overlay brain marks that folder in use at start, so
a later `nell update` prune doesn't delete it from under the running process."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

import brain
from brain.bridge.server import build_app
from brain.update import overlay


def test_lifespan_marks_the_loaded_overlay_in_use(persona_dir: Path, tmp_path: Path, monkeypatch):
    folder = overlay.overlay_root() / "aaaa"  # conftest: KINDLED_HOME under tmp_path
    (folder / "brain").mkdir(parents=True)
    monkeypatch.setattr(brain, "__file__", str(folder / "brain" / "__init__.py"))
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert (folder / overlay.IN_USE_DIR / str(os.getpid())).is_file()
