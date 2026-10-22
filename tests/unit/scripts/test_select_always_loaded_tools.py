"""Tests for scripts/select_always_loaded_tools.py — the #525 rule that picks which brain tools stay up front."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("select_always_loaded_under_test", REPO / "scripts" / "select_always_loaded_tools.py")
sel = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sel
_spec.loader.exec_module(sel)


def test_wilson_lower_bound_matches_the_recorded_selection():
    # usage_selection.out.txt on nell's logs: record_monologue 117 of 168 chat turns, lower bound 62.3%.
    assert round(sel._wilson_lo(117, 168), 3) == 0.623
    assert sel._wilson_lo(0, 0) == 0.0


def _persona(tmp_path, monologue_turns, n=40):
    p = tmp_path / "devpersona"
    p.mkdir()
    chats, calls = [], []
    for i in range(n):
        ts = f"2026-07-01T10:{i:02d}:30+00:00"
        chats.append({"call_type": "chat", "ts": ts, "num_turns": 2, "cache_read_input_tokens": 40000})
        if i < monologue_turns:
            calls.append({"name": "record_monologue", "timestamp": f"2026-07-01T10:{i:02d}:10+00:00"})
    calls.append({"name": "read_file", "timestamp": "2026-07-01T10:00:10+00:00"})
    (p / "chat_usage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chats), encoding="utf-8")
    (p / "tool_invocations.log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in calls), encoding="utf-8")
    return p


def test_without_a_captured_body_the_schemas_come_from_the_brain(tmp_path, capsys):
    persona = _persona(tmp_path, monologue_turns=30)
    assert sel.main([str(persona), "--from", "2026-06-01", "--to", "2026-08-01"]) == 0
    out = capsys.readouterr().out
    assert "ALWAYS_LOAD = ['record_monologue']" in out
    assert "read_file" in out and "-> defer" in out.split("read_file", 1)[1].splitlines()[0]


def test_a_rarely_used_tool_is_deferred(tmp_path, capsys):
    persona = _persona(tmp_path, monologue_turns=0)
    sel.main([str(persona), "--from", "2026-06-01", "--to", "2026-08-01"])
    assert "ALWAYS_LOAD = []" in capsys.readouterr().out


def test_the_window_must_be_given(tmp_path):
    # Round 2 N-4: a stale default window would reprint the old selection and look like a confirmation.
    persona = _persona(tmp_path, monologue_turns=30)
    with pytest.raises(SystemExit) as exc:
        sel.main([str(persona)])
    assert exc.value.code == 2
