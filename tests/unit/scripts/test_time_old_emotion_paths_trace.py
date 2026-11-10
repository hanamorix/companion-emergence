"""scripts/time_old_emotion_paths.py - the trace timing still runs against write_trace_memory's current signature.

The script times one trace write on a scratch copy of a store. It catches only ImportError/AttributeError, so a caller
left on an old signature would crash the run with a TypeError rather than report a skip.
"""

from __future__ import annotations

import importlib.util
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
    assert any("write_trace_memory, one trace of 13 characters" in line for line in lines)
