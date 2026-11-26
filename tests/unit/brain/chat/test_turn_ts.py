"""#390 / #598 — the user's message is stamped once, when the turn starts.

Driven through ``engine.respond`` with the real ``ClaudeCliProvider`` (``subprocess.run`` patched to
capture the stdin it would pipe to ``claude``), the same path as ``test_context_narrative_path.py``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.bridge.provider import ClaudeCliProvider
from brain.chat.engine import respond
from brain.chat.session import create_session, reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def _persona(tmp_path: Path) -> Path:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    cfg = {"provider": "fake", "searcher": "noop", "user_name": "Hana"}
    (persona_dir / "persona_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return persona_dir


def _cli_result() -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "ok"})
    m.stderr = ""
    return m


def _turns(tmp_path: Path, texts: list[str]) -> list[str]:
    """Run one engine turn per text in one session; return the stdin each turn piped to ``claude``."""
    reset_registry()
    persona_dir = _persona(tmp_path)
    session = create_session(persona_dir.name)
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    prompts: list[str] = []
    try:
        for text in texts:
            with patch("brain.bridge.provider.subprocess.run", return_value=_cli_result()) as mock_run:
                respond(
                    persona_dir,
                    text,
                    store=store,
                    hebbian=hebbian,
                    provider=ClaudeCliProvider(probe_model="sonnet"),
                    session=session,
                    voice_md_override="# Nell",
                )
            chat_calls = [c for c in mock_run.call_args_list if c.kwargs.get("input")]
            prompts.append(chat_calls[-1].kwargs["input"])
    finally:
        store.close()
        hebbian.close()
        reset_registry()
    return prompts


def _hana_lines(prompt: str) -> list[str]:
    return [line for line in prompt.split("\n") if re.match(r"^(?:\S+ )?Hana: ", line)]


def test_390_the_live_user_line_reaches_the_next_turn_byte_identical(tmp_path: Path) -> None:
    # a first turn with no history is sent as the bare message, so compare turn 2's live line with turn 3
    _, second, third = _turns(tmp_path, ["first message", "second message", "third message"])
    live = _hana_lines(second)[-1]
    replayed = _hana_lines(third)[1]
    assert live.endswith("Hana: second message")
    assert replayed == live


class _Clock(datetime):
    """The buffer's clock: reads ``SENT`` until the provider is called, ``REPLIED`` after."""

    current = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):  # noqa: ANN001, ANN206
        return cls.current


def test_598_the_stored_user_row_carries_the_time_it_was_sent(tmp_path: Path) -> None:
    sent = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    replied = datetime(2026, 1, 1, 9, 5, tzinfo=UTC)
    _Clock.current = sent

    def slow_reply(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        _Clock.current = replied
        return _cli_result()

    reset_registry()
    persona_dir = _persona(tmp_path)
    session = create_session(persona_dir.name)
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        with (
            patch("brain.ingest.buffer.datetime", _Clock),
            patch("brain.bridge.provider.subprocess.run", side_effect=slow_reply),
        ):
            respond(
                persona_dir,
                "hello",
                store=store,
                hebbian=hebbian,
                provider=ClaudeCliProvider(probe_model="sonnet"),
                session=session,
                voice_md_override="# Nell",
            )
    finally:
        store.close()
        hebbian.close()
        reset_registry()
    buf = persona_dir / "active_conversations" / f"{session.session_id}.jsonl"
    rows = {r["speaker"]: r for r in map(json.loads, buf.read_text(encoding="utf-8").splitlines())}
    assert rows["user"]["ts"] == sent.isoformat(timespec="seconds")
    assert rows["assistant"]["ts"] == replied.isoformat(timespec="seconds")
