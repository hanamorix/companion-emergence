"""The S0 stub engine's `batch()` (Engine 0, #575).

`engine.engine_for(persona_dir).batch()` is a context manager on the engine instance (spec 3.0 batch scope).
In the stub it does nothing: it takes no argument, yields None, records nothing, writes nothing and lets an
exception from the block go out unchanged. The engine built later (S4) collects the block's events and applies
them once; its own tests replace the "does nothing" ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.emotion.engine import EmotionEngine


def test_batch_is_a_context_manager_that_yields_none(tmp_path: Path) -> None:
    engine = EmotionEngine(tmp_path)
    with engine.batch() as value:
        assert value is None


def test_batch_takes_no_arguments(tmp_path: Path) -> None:
    engine = EmotionEngine(tmp_path)
    with pytest.raises(TypeError):
        engine.batch(1)  # type: ignore[call-arg]


def test_batch_does_nothing(tmp_path: Path) -> None:
    engine = EmotionEngine(tmp_path)
    with engine.batch():
        pass
    assert engine.recorded_calls == []
    assert list(tmp_path.iterdir()) == []


def test_nested_batches_are_harmless(tmp_path: Path) -> None:
    engine = EmotionEngine(tmp_path)
    with engine.batch():
        with engine.batch():
            pass
        with engine.batch():
            pass
    assert engine.recorded_calls == []
    assert list(tmp_path.iterdir()) == []


def test_batch_lets_an_exception_propagate(tmp_path: Path) -> None:
    engine = EmotionEngine(tmp_path)
    raised = ValueError("from the block")
    with pytest.raises(ValueError) as excinfo:
        with engine.batch():
            raise raised
    assert excinfo.value is raised
    assert engine.recorded_calls == []
