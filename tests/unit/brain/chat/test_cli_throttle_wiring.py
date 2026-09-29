"""engine.respond() must call cli_throttle.note_user_message/note_reply_end on
every turn (ram-spike-fix INC-6: replaces the retired mark_interactive_active
call sites in engine.py — mark_interactive_active itself is KEPT in
cli_throttle.py as a test/back-compat helper, but production code no longer
calls it)."""

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
    """engine.respond must stamp chat activity (note_user_message) so
    background CLI yields."""
    import brain.bridge.cli_throttle as throttle

    called: dict[str, int] = {"n": 0}
    monkeypatch.setattr(
        throttle,
        "note_user_message",
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

    assert called["n"] >= 1, "note_user_message was not called by respond()"


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
    """respond() must stamp activity at turn-END (note_reply_end) as well as
    turn-START (note_user_message) — ram-spike-fix INC-6's exception-safe
    split of the old single mark_interactive_active-twice shape (3376b2c1's
    end-of-turn re-stamp rationale, preserved).

    A long LLM call (e.g. 5-minute tool round-trip) can exhaust the idle
    window before the turn finishes, letting a background job fire concurrently.
    The end-of-turn stamp (now in a `finally`, via the `respond` wrapper)
    resets the idle window to turn-END even on an exception.
    """
    import brain.bridge.cli_throttle as throttle

    called: dict[str, int] = {"start": 0, "end": 0}
    monkeypatch.setattr(
        throttle,
        "note_user_message",
        lambda *a, **k: called.__setitem__("start", called["start"] + 1),
    )
    monkeypatch.setattr(
        throttle,
        "note_reply_end",
        lambda *a, **k: called.__setitem__("end", called["end"] + 1),
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

    assert called["start"] >= 1 and called["end"] >= 1, (
        f"note_user_message called {called['start']}x, note_reply_end called "
        f"{called['end']}x; expected >=1 each (start + end of turn)"
    )


def test_respond_calls_note_reply_end_even_when_inner_body_raises(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exception-safety point of the respond()/_respond_inner() split
    (ram-spike-fix INC-6): note_reply_end() must fire in a `finally` even
    when the turn's own body raises, so the in-flight counter is never left
    stuck (which would make is_chat_idle() report False forever)."""
    import brain.chat.engine as engine_mod
    from brain.bridge import cli_throttle

    monkeypatch.setattr(
        engine_mod,
        "_respond_inner",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    cli_throttle.reset()
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        with pytest.raises(RuntimeError):
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

    assert cli_throttle._inflight_replies == 0, (  # noqa: SLF001 — the exact invariant under test
        "an exception in the turn body must not leave the in-flight counter stuck"
    )
