"""scripts/time_old_emotion_paths.py - the trace timing still runs against write_trace_memory's current signature.

The script times one trace write on a scratch copy of a store. It catches only ImportError/AttributeError, so a caller
left on an old signature would crash the run with a TypeError rather than report a skip.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "time_old_emotion_paths.py"

_spec = importlib.util.spec_from_file_location("time_old_emotion_paths_trace_under_test", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


@pytest.mark.no_base_failure(reason="base script matches base signature; bites when the writer's signature moves on")
def test_time_trace_runs_on_a_temp_store(tmp_path: Path):
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    try:
        store.create(Memory.create_new(content="a quiet drift", memory_type="monologue_trace", domain="monologue"))
        lines = mod._time_trace(tmp_path, store)
    finally:
        store.close()
    # No start step: the single timing, labelled as the switch-off path.
    assert [line for line in lines if line.startswith("write_trace_memory")] == [
        line for line in lines if "switch-off path (persona_dir=None" in line and "one trace of 13 characters" in line
    ]
    assert sum(line.startswith("write_trace_memory") for line in lines) == 1
    assert not any("LIVE path" in line for line in lines)


def _started_persona(root: Path):
    from brain.memory.store import Memory, MemoryStore

    root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(root / "memories.db")
    store.create(Memory.create_new(content="a quiet drift", memory_type="monologue_trace", domain="monologue"))
    mod._step_start(root)
    return store


def test_a_started_copy_times_the_live_path_and_the_switch_off_path_each_labelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from brain.monologue import trace

    calls: list[object] = []
    real = trace.write_trace_memory

    def spy(store, text, **kwargs):
        calls.append(kwargs["persona_dir"])
        return real(store, text, **kwargs)

    spy.__signature__ = inspect.signature(real)  # the script reads the writer's signature to pick its keyword arguments
    monkeypatch.setattr(trace, "write_trace_memory", spy)
    persona = tmp_path / "persona"
    store = _started_persona(persona)
    try:
        lines = mod._time_trace(persona, store, True)
    finally:
        store.close()
    from brain.emotion.engine import engine_for

    assert engine_for(persona).is_live()  # so the first ten calls take the live branch of the writer
    heads = [line for line in lines if line.startswith("write_trace_memory")]
    assert len(heads) == 2
    assert "LIVE path (persona_dir=the copy, engine live" in heads[0]
    assert "switch-off path (persona_dir=None" in heads[1]
    # the first timing's ten calls pass the copy, the second's ten pass None
    assert calls == [persona] * 10 + [None] * 10


def test_a_commit_without_ensure_engine_ready_has_no_start_step(monkeypatch: pytest.MonkeyPatch):
    from brain.emotion import engine

    assert mod._has_start_step() is True
    monkeypatch.delattr(engine, "ensure_engine_ready")
    assert mod._has_start_step() is False


def test_the_load_line_prints_the_one_minute_average_labelled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod.os, "getloadavg", lambda: (0.5, 0.7, 0.9), raising=False)
    assert mod._load_line("at start") == "load average (1 min), at start: 0.50"
    assert mod._load_line("at end") == "load average (1 min), at end: 0.50"


def test_the_load_line_says_unknown_where_there_is_no_load_average(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delattr(mod.os, "getloadavg", raising=False)
    assert mod._load_line("at start") == "load average (1 min), at start: unknown"

    def fails():
        raise OSError("no load average")

    monkeypatch.setattr(mod.os, "getloadavg", fails, raising=False)
    assert mod._load_line("at end") == "load average (1 min), at end: unknown"
