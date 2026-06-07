"""engine.respond() must call cli_throttle.mark_interactive_active on every turn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.chat.engine import respond
from brain.chat.session import reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _reset_sessions():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}),
        encoding="utf-8",
    )
    return d


def test_respond_marks_interactive_active(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """engine.respond must mark interactive-active so background CLI yields."""
    import brain.bridge.cli_throttle as throttle

    called: dict[str, int] = {"n": 0}
    monkeypatch.setattr(
        throttle,
        "mark_interactive_active",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )

    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        respond(
            persona_dir,
            "hello",
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            voice_md_override="# Nell\n\nHello.",
        )
    finally:
        store.close()
        hebbian.close()

    assert called["n"] >= 1, "mark_interactive_active was not called by respond()"
