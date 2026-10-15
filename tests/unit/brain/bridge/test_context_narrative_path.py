"""#227 — the narrative history reaches the model through the REAL provider path.

``test_context_narrative.py`` pins the renderer. These tests drive ``engine.respond`` with the real
``ClaudeCliProvider`` (``subprocess.run`` / ``Popen`` patched to capture the stdin it would send), so the
option plumbing (``chat_options["user_name"]`` -> ``chat`` / ``chat_stream`` -> formatter), the system-message
split and the volatile suffix are all exercised, not re-implemented in the test.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import ClaudeCliProvider
from brain.chat.engine import respond
from brain.chat.session import create_session, reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

_REPLY_FRAME_TAIL = "longer when the answer needs room."


def _persona(tmp_path: Path, *, user_name: str | None) -> Path:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    cfg: dict = {"provider": "fake", "searcher": "noop"}
    if user_name is not None:
        cfg["user_name"] = user_name
    (persona_dir / "persona_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return persona_dir


def _seed_history(persona_dir: Path, session_id: str) -> None:
    rows = [
        {
            "session_id": session_id,
            "speaker": "user",
            "text": "hey nell",
            "ts": "2026-05-20T10:00:00Z",
        },
        {
            "session_id": session_id,
            "speaker": "assistant",
            "text": "i am here",
            "ts": "2026-05-20T10:00:30Z",
        },
        {
            "session_id": session_id,
            "speaker": "user",
            "text": "two\nlines",
            "ts": "2026-05-20T10:05:00Z",
        },
        {
            "session_id": session_id,
            "speaker": "assistant",
            "text": "noted",
            "ts": "2026-05-20T10:05:20Z",
        },
    ]
    d = persona_dir / "active_conversations"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{session_id}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _cli_result() -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "result": "ok"}
    )
    m.stderr = ""
    return m


def _run_turn(tmp_path: Path, *, user_name: str | None) -> str:
    """One engine turn through the real ClaudeCliProvider; return the stdin it piped to ``claude``."""
    reset_registry()
    persona_dir = _persona(tmp_path, user_name=user_name)
    session = create_session(persona_dir.name)
    _seed_history(persona_dir, session.session_id)
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        with patch("brain.bridge.provider.subprocess.run", return_value=_cli_result()) as mock_run:
            respond(
                persona_dir,
                "what do you remember?",
                store=store,
                hebbian=hebbian,
                provider=ClaudeCliProvider(model="sonnet"),
                session=session,
                voice_md_override="# Nell",
            )
    finally:
        store.close()
        hebbian.close()
        reset_registry()
    chat_calls = [c for c in mock_run.call_args_list if c.kwargs.get("input")]
    assert chat_calls, "the provider never piped a prompt to claude"
    return chat_calls[-1].kwargs["input"]


def test_c2a_respond_sends_narrative_lines_named_after_the_configured_user(tmp_path: Path) -> None:
    prompt = _run_turn(tmp_path, user_name="Hana")

    said = [line for line in prompt.split("\n") if re.match(r"^(?:\S+ )?(?:Hana|you): ", line)]
    assert len(said) == 5  # four seeded turns plus the new user message
    assert re.match(r"^\S+ Hana: hey nell$", said[0])
    assert re.match(r"^\S+ you: i am here$", said[1])
    assert re.match(r"^\S+ Hana: two\\nlines$", said[2])  # one line: the newline stays escaped
    assert said[4] == "Hana: what do you remember?"  # the live message has no time yet (stamped at persist)
    assert not any(line.startswith("{") for line in prompt.split("\n"))
    assert '"speaker"' not in prompt and "JSONL" not in prompt
    assert "`ts`" not in prompt  # the ambient tail no longer describes a field
    assert "The time at the start of each conversation line above" in prompt
    # the volatile suffix is last and the reply frame is last inside it
    assert prompt.rstrip().endswith(_REPLY_FRAME_TAIL)
    assert prompt.index("Answer the final human entry directly") < prompt.index(said[0])


def test_c2c_without_a_configured_name_the_lines_say_user(tmp_path: Path) -> None:
    prompt = _run_turn(tmp_path, user_name=None)
    said = [line for line in prompt.split("\n") if re.match(r"^(?:\S+ )?(?:User|you): ", line)]
    assert said[0].endswith(" User: hey nell")
    assert said[4] == "User: what do you remember?"


class _KeepStdin(io.StringIO):
    """A stdin that remembers what was written before ``chat_stream`` closes it."""

    final: str = ""

    def close(self) -> None:
        self.final = self.getvalue()
        super().close()


class _StreamPopen:
    def __init__(self) -> None:
        self.stdin = _KeepStdin()
        self.stdout = iter(
            [json.dumps({"type": "result", "is_error": False, "result": "ok"}) + "\n"]
        )
        self.stderr = io.StringIO("")
        self.pid = 4242

    def wait(self, timeout=None):  # noqa: ANN001, ANN201
        return 0

    def poll(self):  # noqa: ANN201
        return 0

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def test_c2b_chat_stream_sends_the_same_narrative_lines() -> None:
    fake = _StreamPopen()
    msgs = [
        ChatMessage(role="system", content="SYSTEM"),
        ChatMessage(role="user", content="hey nell", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="i am here", ts="2026-05-20T10:00:30Z"),
        ChatMessage(role="user", content="and now?"),
    ]
    with patch("subprocess.Popen", return_value=fake):
        list(
            ClaudeCliProvider(model="sonnet").chat_stream(
                msgs,
                options={
                    "user_name": "Hana",
                    "volatile_suffix": "SUFFIX",
                    "include_block_clock": False,
                },
            )
        )
    sent = fake.stdin.final
    said = [line for line in sent.split("\n") if re.match(r"^(?:\S+ )?(?:Hana|you): ", line)]
    assert len(said) == 3
    assert said[0].endswith(" Hana: hey nell") or said[0] == "Hana: hey nell"
    assert said[2] == "Hana: and now?"
    assert sent.endswith("\n\nSUFFIX")
    assert "SYSTEM" not in sent  # the first system message is split off into the system prompt
    assert "Conversation context is encoded" not in sent


@pytest.mark.parametrize("name", [None, "", "   "])
def test_c2_blank_names_fall_back_to_user(name: str | None) -> None:
    from brain.bridge.provider import _format_claude_print_prompt

    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
        ChatMessage(role="user", content="c"),
    ]
    out = _format_claude_print_prompt(msgs, user_name=name)
    assert "User: a" in out and "User: c" in out
