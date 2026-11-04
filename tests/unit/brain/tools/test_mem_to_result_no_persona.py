"""A tool result built with no persona directory carries the stored emotions and asks no engine.

The search tool and `read_full_memory` build their result with `_mem_to_result`. A caller that has no persona directory
(the recall-diagnostics emission test passes None on purpose) has no engine to ask, so the result takes the stored values.
The engine stub answers different (faded) values and records every `engine_for` call, so these tests reject a fix that
still calls `engine_for` and swallows the error: the stored values would not come back and the call count would not be zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from brain.emotion import engine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls._common import _mem_to_result
from brain.tools.impls.read_full_memory import read_full_memory

_STORED = {"love": 7.2, "grief": 3.4}


def _record_engine_for_calls(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Wrap whatever `engine.engine_for` is now (the stub's) in a recorder; call this AFTER any memory is seeded."""
    calls: list[Any] = []
    inner = engine.engine_for

    def _recorder(persona_dir: Any) -> Any:
        calls.append(persona_dir)
        return inner(persona_dir)

    monkeypatch.setattr(engine, "engine_for", _recorder)
    return calls


def test_a_result_with_no_persona_carries_the_stored_emotions_and_asks_no_engine(
    monkeypatch: pytest.MonkeyPatch, emotion_engine_stub
) -> None:
    memory = Memory.create_new(content="a stored memory", memory_type="episodic", domain="us", emotions=dict(_STORED))
    emotion_engine_stub.faded = {(memory.id, "love"): 4.0, (memory.id, "grief"): 1.0}
    calls = _record_engine_for_calls(monkeypatch)

    result = _mem_to_result(memory, None)

    assert result["emotions"] == _STORED
    assert calls == []


def test_a_result_with_no_persona_does_not_raise() -> None:
    """No stub: the real `engine_for` is what a missing directory used to reach (it raised on None)."""
    memory = Memory.create_new(content="a stored memory", memory_type="episodic", domain="us", emotions=dict(_STORED))

    assert _mem_to_result(memory, None)["emotions"] == _STORED


def test_read_full_memory_with_no_persona_carries_the_stored_emotions_and_asks_no_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, emotion_engine_stub
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    try:
        memory = Memory.create_new(
            content="a stored memory", memory_type="episodic", domain="us", emotions=dict(_STORED)
        )
        store.create(memory)
        emotion_engine_stub.faded = {(memory.id, "love"): 4.0, (memory.id, "grief"): 1.0}
        calls = _record_engine_for_calls(monkeypatch)  # after the seeding: `create` asks the store's engine

        result = read_full_memory(
            memory.id,
            store=store,
            hebbian=HebbianMatrix(tmp_path / "hebbian.db"),
            persona_dir=None,  # type: ignore[arg-type]
        )

        assert result["emotions"] == _STORED
        assert calls == []
    finally:
        store.close()
