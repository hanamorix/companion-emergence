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


def test_respond_degrades_to_full_suite_when_salience_raises(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If assess_salience raises, respond() must not crash — degrade to full suite.

    The guard wraps the salience+recruit computation: an exception there
    must result in allowed=None being passed to build_tools_list (full suite)
    and signal=None to run_tool_loop, rather than propagating out of respond().
    """
    import brain.chat.engine as engine_mod

    def _boom(*_a, **_kw):
        raise RuntimeError("salience exploded")

    monkeypatch.setattr(engine_mod, "assess_salience", _boom)

    # Capture what allowed value reaches build_tools_list.
    original_build = engine_mod.build_tools_list
    captured: dict[str, object] = {}

    def _spy_build(**kwargs):
        captured["allowed"] = kwargs.get("allowed")
        return original_build(**kwargs)

    monkeypatch.setattr(engine_mod, "build_tools_list", _spy_build)

    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        result = respond(
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

    # respond() must complete without raising.
    assert result is not None
    # build_tools_list must have been called with allowed=None (full suite fallback).
    assert captured.get("allowed") is None, (
        f"expected allowed=None for full-suite fallback, got {captured.get('allowed')!r}"
    )


def test_respond_re_stamps_interactive_active_at_turn_end(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """respond() must call mark_interactive_active at turn-END as well as turn-START.

    A long LLM call (e.g. 5-minute tool round-trip) can exhaust the idle
    window before the turn finishes, letting a background job fire concurrently.
    The second stamp just before ChatResult is returned resets the idle window
    to turn-END.
    """
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

    assert called["n"] >= 2, (
        f"mark_interactive_active was called {called['n']} time(s); "
        "expected ≥2 (start + end of turn)"
    )
