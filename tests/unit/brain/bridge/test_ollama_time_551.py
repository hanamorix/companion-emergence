"""#551 — an Ollama chat turn carries the per-turn volatile text and each user message's send time.

The Claude CLI path renders both into its stdin prompt. ``OllamaProvider`` used to strip ``volatile_suffix`` with the
other routing keys and send history through ``ChatMessage.to_dict()``, which has no time, so a kindled on a local
model never knew the time or how long it had been since anyone spoke. Times go on the user's messages only: the
kindled's own replies are sent as they are, so a small model never sees itself writing a timestamp.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import OllamaProvider, _narrative_time

T1 = "2026-05-20T10:00:00Z"
T2 = "2026-05-20T10:00:30Z"
T3 = "2026-05-21T08:15:00Z"
SUFFIX = "[current time: NOW]\nNOTE"


def _ollama_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"message": {"role": "assistant", "content": "ok"}, "done": True}
    resp.raise_for_status = MagicMock()
    return resp


def _history() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="SYS"),
        ChatMessage(role="user", content="hey nell", ts=T1),
        ChatMessage(role="assistant", content="i am here", ts=T2),
        ChatMessage(role="user", content="no time"),
        ChatMessage(role="user", content="bad time", ts="not-a-time"),
        ChatMessage(role="user", content="and now?", ts=T3),
    ]


_OPTIONS = {
    "volatile_suffix": SUFFIX,
    "include_block_clock": False,
    "session_id": "s",
    "temperature": 0.4,
}


def test_551_chat_sends_the_volatile_text_and_each_user_messages_time() -> None:
    with patch("httpx.post", return_value=_ollama_response()) as mock_post:
        OllamaProvider().chat(_history(), options=dict(_OPTIONS))

    payload = mock_post.call_args.kwargs["json"]
    sent = payload["messages"]
    assert sent[0] == {"role": "system", "content": "SYS"}
    assert sent[1] == {"role": "user", "content": f"{_narrative_time(T1)} hey nell"}
    assert sent[2] == {"role": "assistant", "content": "i am here"}
    assert sent[3] == {"role": "user", "content": "no time"}
    assert sent[4] == {"role": "user", "content": "bad time"}
    assert sent[5] == {"role": "user", "content": f"{_narrative_time(T3)} and now?\n\n{SUFFIX}"}
    assert len(sent) == 6
    assert payload["options"] == {"temperature": 0.4}


def _stream_cm() -> MagicMock:
    response = MagicMock()
    response.iter_lines = MagicMock(return_value=iter(['{"message": {"content": "ok"}, "done": true}']))
    response.raise_for_status = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=response)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_551_chat_stream_sends_the_same_messages_as_chat() -> None:
    with patch("httpx.post", return_value=_ollama_response()) as mock_post:
        OllamaProvider().chat(_history(), options=dict(_OPTIONS))
    with patch("httpx.stream", return_value=_stream_cm()) as mock_stream:
        assert list(OllamaProvider().chat_stream(_history(), options=dict(_OPTIONS))) == ["ok"]

    streamed = mock_stream.call_args.kwargs["json"]
    assert streamed["messages"] == mock_post.call_args.kwargs["json"]["messages"]
    assert streamed["messages"][-1]["content"].endswith(f"\n\n{SUFFIX}")
    assert streamed["options"] == {"temperature": 0.4}


def test_551_the_suffix_lands_once_on_the_latest_user_message_in_a_tool_loop_follow_up() -> None:
    from brain.bridge.chat import ToolCall

    messages = [
        ChatMessage(role="system", content="SYS"),
        ChatMessage(role="user", content="what did we say?", ts=T1),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="c1", name="search_memories", arguments={"query": "x"}),),
        ),
        ChatMessage(role="tool", content="found it", tool_call_id="c1"),
    ]
    with patch("httpx.post", return_value=_ollama_response()) as mock_post:
        OllamaProvider().chat(messages, options={"volatile_suffix": SUFFIX})
        OllamaProvider().chat(messages, options={"volatile_suffix": SUFFIX})

    first, second = (c.kwargs["json"]["messages"] for c in mock_post.call_args_list)
    assert first == second  # a follow-up call sends the suffix once, never stacked
    assert first[1]["content"] == f"{_narrative_time(T1)} what did we say?\n\n{SUFFIX}"
    assert first[3] == messages[3].to_dict()
    assert first[2] == messages[2].to_dict()


def test_551_with_no_user_message_the_suffix_is_sent_as_a_trailing_user_message() -> None:
    with patch("httpx.post", return_value=_ollama_response()) as mock_post:
        OllamaProvider().chat([ChatMessage(role="system", content="SYS")], options={"volatile_suffix": SUFFIX})

    assert mock_post.call_args.kwargs["json"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": SUFFIX},
    ]


def test_551_an_image_user_message_gets_the_time_and_suffix_as_text_blocks() -> None:
    from brain.bridge.chat import ImageBlock, TextBlock

    image = ImageBlock(image_sha="a" * 64, media_type="image/png")
    msg = ChatMessage(role="user", content=(TextBlock(text="look"), image), ts=T1)
    with patch("httpx.post", return_value=_ollama_response()) as mock_post:
        OllamaProvider().chat([msg], options={"volatile_suffix": SUFFIX})

    content = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": _narrative_time(T1)},
        *msg.to_dict()["content"],
        {"type": "text", "text": SUFFIX},
    ]


def test_551_respond_through_ollama_carries_the_clock_and_user_times(tmp_path) -> None:
    import json

    from brain.chat.engine import respond
    from brain.chat.session import create_session, reset_registry
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    reset_registry()
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8"
    )
    session = create_session(persona_dir.name)
    rows = [
        {"session_id": session.session_id, "speaker": "user", "text": "hey nell", "ts": T1},
        {"session_id": session.session_id, "speaker": "assistant", "text": "i am here", "ts": T2},
    ]
    active = persona_dir / "active_conversations"
    active.mkdir()
    (active / f"{session.session_id}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        with patch("httpx.post", return_value=_ollama_response()) as mock_post:
            respond(
                persona_dir,
                "what do you remember?",
                store=store,
                hebbian=hebbian,
                provider=OllamaProvider(),
                session=session,
                voice_md_override="# Nell",
            )
    finally:
        store.close()
        hebbian.close()
        reset_registry()

    chat_calls = [c.kwargs["json"] for c in mock_post.call_args_list if c.kwargs["json"].get("tools")]
    assert chat_calls, "respond() never made its chat call"
    sent = chat_calls[0]["messages"]
    users = [m for m in sent if m["role"] == "user"]
    assert users[0]["content"] == f"{_narrative_time(T1)} hey nell"
    assert [m["content"] for m in sent if m["role"] == "assistant"] == ["i am here"]
    latest = users[-1]["content"]
    assert "what do you remember?" in latest
    assert "[current time: " in latest
    assert "The time at the start of each conversation line above" in latest
    assert "volatile_suffix" not in chat_calls[0].get("options", {})
