"""`nell chat --no-bridge`'s REPL starts each turn with the "bridge timed out" flag cleared (#803 step 9).

One transient timeout of the bridge must not put a whole interactive session on the read-only vocabulary file: the REPL is the one
long-lived process that makes vocabulary calls and is not the bridge, so it clears the flag itself at the top of every turn.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import cli as brain_cli
from brain.emotion import vocabulary_access


def test_each_repl_turn_starts_with_the_timeout_flag_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import brain.chat.engine as chat_engine
    import brain.chat.session as chat_session
    import brain.ingest.pipeline as ingest_pipeline
    from brain.memory import floor_startup

    persona = tmp_path / "persona"
    persona.mkdir()
    monkeypatch.setattr(brain_cli, "get_persona_dir", lambda name: persona)
    monkeypatch.setattr(brain_cli, "_resolve_routing", lambda persona_dir, args: ("fake", "fake"))
    monkeypatch.setattr(brain_cli, "_start_floor_bootstrap_for_direct_chat", lambda persona_dir, *, blocking: None)
    monkeypatch.setattr(brain_cli, "_engine_ready_or_refuse", lambda persona_dir, dry_run: None)
    monkeypatch.setattr(brain_cli, "get_provider", lambda name: object())
    monkeypatch.setattr(brain_cli, "_drain_emotion_engine", lambda persona_dir: None)
    monkeypatch.setattr(brain_cli, "_flush_cost_series_at_exit", lambda store: None)
    monkeypatch.setattr(brain_cli, "_drain_pass2_at_exit", lambda persona_dir: None)
    monkeypatch.setattr(vocabulary_access, "load_read_only", lambda persona_dir: None)
    monkeypatch.setattr(floor_startup, "start_requested_ram_figure", lambda: None)
    monkeypatch.setattr(chat_session, "create_session", lambda name: SimpleNamespace(session_id="s", turns=0))
    monkeypatch.setattr(ingest_pipeline, "extract_session_snapshot", lambda *a, **k: None)

    seen_at_turn_start: list[bool | None] = []

    def respond(persona_dir, message, **_k):
        seen_at_turn_start.append(vocabulary_access._BRIDGE_TIMED_OUT.get(persona))
        vocabulary_access._BRIDGE_TIMED_OUT[persona] = True  # this turn's bridge call timed out
        return SimpleNamespace(content="ok", tool_invocations=[])

    monkeypatch.setattr(chat_engine, "respond", respond)
    lines = iter(["first", "second", "third"])

    def fake_input(prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    vocabulary_access._BRIDGE_TIMED_OUT[persona] = True  # left over from before the session's first turn
    args = SimpleNamespace(persona="p", message=None, close=False, provider=None, model=None, dry_run=False)
    assert brain_cli._chat_direct_mode(args) == 0
    assert seen_at_turn_start == [False, False, False]
