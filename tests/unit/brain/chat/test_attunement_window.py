from brain.bridge.chat import ChatMessage
from brain.chat.tool_loop import _ATTUNEMENT_WINDOW, _buffer_slice_from_messages


def test_buffer_slice_capped_to_window():
    msgs = [ChatMessage(role="user", content=f"turn {i}") for i in range(30)]
    sl = _buffer_slice_from_messages(msgs)
    assert len(sl) == _ATTUNEMENT_WINDOW
    assert sl[-1].content == "turn 29"  # keeps the most recent


def test_buffer_slice_under_window_returns_all():
    msgs = [ChatMessage(role="user", content=f"turn {i}") for i in range(3)]
    sl = _buffer_slice_from_messages(msgs)
    assert len(sl) == 3


def test_buffer_slice_skips_empty_and_nonuser():
    msgs = [
        ChatMessage(role="user", content="real"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="   "),  # whitespace-only — skipped
    ]
    sl = _buffer_slice_from_messages(msgs)
    assert len(sl) == 1 and sl[0].content == "real"
