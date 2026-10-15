"""#399: a `--max-budget-usd` overrun makes the real CLI exit 1 (not 0), so the blocking paths' graceful
reply was unreachable.

`ClaudeCliProvider.chat()` (text branch) and `_chat_with_mcp_tools` both had a graceful path for an
over-budget result frame, but `_run_blocking_cli` raised `claude_cli_exit` on the non-zero exit before
anyone looked at the frame. Every earlier budget test stubbed exit 0, so it stayed green. These tests
replay the REAL frames at the REAL exit code.

The fixture is the stdout a claude 2.1.300 run produced under a $0.0001 budget (`stream-json --verbose`,
exit 1): its `assistant` and `result` frames as captured, with the `init` frame (home paths, plugin and
skill lists of the capturing machine) reduced to the four-key shape `cli_2_1_284_web_tools.ndjson` uses
and the hook frames dropped. The real `result` frame has `subtype` `error_max_budget_usd`, `is_error`, an
`errors` list, `terminal_reason` `budget_exhausted` and NO `result` key; its `usage` token fields are 0
(the real numbers are in `modelUsage`) while `total_cost_usd` is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.bridge.provider import _BUDGET_EXCEEDED_MSG, ProviderError
from tests.bridge.test_provider_builtin_tool_audit import (
    _AUTH_TEXT,
    _blocking,
    _lines,
    _mcp_row_during_run,
    _persona,
    _rows,
    _stream,
    _text_branch,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "cli_2_1_300_budget_exceeded.ndjson"


def _stdout() -> str:
    return _FIXTURE.read_text(encoding="utf-8")


def _result_frame() -> dict:
    return json.loads(_stdout().strip().split("\n")[-1])


def test_the_fixture_is_the_real_over_budget_shape():
    frame = _result_frame()
    assert frame["type"] == "result"
    assert frame["subtype"] == "error_max_budget_usd" and frame["is_error"] is True
    assert frame["terminal_reason"] == "budget_exhausted"
    assert "result" not in frame
    assert any("maximum budget" in e.lower() for e in frame["errors"])
    assert frame["total_cost_usd"] == 0.0019682
    assert not [h for h in _stdout().split("\n") if '"hook_' in h], "hook frames must stay out"
    assert "/Users/" not in _stdout(), "the capturing machine's paths must stay out of the public repo"


def test_tools_branch_over_budget_exit_1_returns_the_graceful_reply(tmp_path):
    persona_dir = _persona(tmp_path)
    resp, run = _blocking(persona_dir, _stdout(), returncode=1)

    assert run.called
    assert resp.content == _BUDGET_EXCEEDED_MSG
    assert resp.raw["subtype"] == "error_max_budget_usd"


def test_text_branch_over_budget_exit_1_returns_the_graceful_reply(tmp_path):
    persona_dir = _persona(tmp_path)
    resp, run, _ = _text_branch(persona_dir, _stdout(), returncode=1)

    assert run.called
    assert resp.content == _BUDGET_EXCEEDED_MSG
    assert resp.raw["subtype"] == "error_max_budget_usd"
    # The fixture's unanswered built-in call is still audited on this (newly reachable) return.
    rows = _rows(persona_dir)
    assert [(r["name"], r["origin"], r["outcome"], r["error"]) for r in rows] == [
        ("WebSearch", "cli_builtin", "error", "no result before the stream ended")
    ]


def _usage_rows(persona_dir: Path) -> list[dict]:
    path = persona_dir / "chat_usage.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def test_over_budget_turn_logs_one_usage_row_per_branch(tmp_path):
    # The streaming path logs the over-budget frame before its budget check; the blocking branches
    # returned first, so the costliest turns (the ones that hit the ceiling) were missing from the log.
    cost = _result_frame()["total_cost_usd"]
    for name, drive in (
        ("tools", lambda d: _blocking(d, _stdout(), returncode=1)),
        ("text", lambda d: _text_branch(d, _stdout(), returncode=1)),
    ):
        persona_dir = _persona(tmp_path / name)
        drive(persona_dir)
        rows = _usage_rows(persona_dir)
        assert len(rows) == 1, (name, rows)
        assert rows[0]["call_type"] == "chat"
        assert rows[0]["total_cost_usd"] == cost
        # The frame's own `usage` is all zeros (the real numbers are in `modelUsage`): the row carries the
        # cost and zero tokens, exactly like the streaming path's row for the same frame.
        assert rows[0]["output_tokens"] == 0 and rows[0]["cache_creation_input_tokens"] == 0


def test_tools_branch_over_budget_reply_keeps_the_turns_dispatched_invocations(tmp_path):
    # The budget is the backstop for runaway tool loops, so an overrun turn has nearly always already run
    # brain tools. Dropping their records dropped image binding, the pass-2 monologue and the turn's
    # tool_invocations out of run_tool_loop (the streaming proxy reads them whatever the reply says).
    persona_dir = _persona(tmp_path)
    resp, _ = _blocking(
        persona_dir, _stdout(), returncode=1, during=_mcp_row_during_run(persona_dir)
    )

    assert resp.content == _BUDGET_EXCEEDED_MSG
    assert [d["name"] for d in resp.dispatched_invocations] == ["search_memories"]


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        ({"subtype": "error_max_budget_usd", "is_error": True, "errors": ["Reached maximum budget ($1)"]}, True),
        ({"subtype": "error_max_budget_usd"}, True),
        ({"is_error": True, "errors": ["Reached maximum budget ($0.75)"]}, True),
        ({"is_error": True, "errors": ["x", None, "REACHED MAXIMUM BUDGET"]}, True),
        ({"is_error": False, "errors": ["Reached maximum budget ($0.75)"]}, False),
        ({"is_error": True, "errors": ["something else"]}, False),
        ({"is_error": True, "result": "Not logged in"}, False),
        ({}, False),
        # Total: a malformed `errors` is "no errors", never an exception.
        ({"is_error": True, "errors": None}, False),
        ({"is_error": True, "errors": 5}, False),
        ({"is_error": True, "errors": True}, False),
        ({"is_error": True, "errors": {"a": "maximum budget"}}, False),
        ({"is_error": True, "errors": "maximum budget"}, False),
    ],
    ids=lambda v: str(v)[:48],
)
def test_is_over_budget_frame_is_exact_and_total(frame, expected):
    from brain.bridge.provider import _is_over_budget_frame

    assert _is_over_budget_frame(frame) is expected


def test_one_definition_of_over_budget_and_every_live_site_calls_it():
    """The predicate was written out five times (one per site, each a chance to drift, and the blocking
    ones never met the real frame). One definition now; this fails if a site grows its own copy or stops
    calling the helper."""
    import ast

    from brain.bridge import provider as provider_mod

    tree = ast.parse(Path(provider_mod.__file__).read_text(encoding="utf-8"))
    docstring_nodes = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    texts = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == "maximum budget" and id(n) not in docstring_nodes
    ]
    assert len(texts) == 1, [n.lineno for n in texts]

    def _calls_helper(function: ast.FunctionDef) -> bool:
        return any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_is_over_budget_frame"
            for n in ast.walk(function)
        )

    found = {}
    for klass in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ClaudeCliProvider"]:
        for fn in [n for n in klass.body if isinstance(n, ast.FunctionDef)]:
            found[f"ClaudeCliProvider.{fn.name}"] = _calls_helper(fn)
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        found[fn.name] = _calls_helper(fn)

    assert sorted(k for k, v in found.items() if v) == [
        "ClaudeCliProvider._chat_with_mcp_tools",
        "ClaudeCliProvider._run_chat_stream",
        "ClaudeCliProvider.chat",
        "_cli_error_detail",
    ]


def _error_frame(**overrides) -> dict:
    return {"type": "result", "subtype": "success", "is_error": True, "result": _AUTH_TEXT, **overrides}


@pytest.mark.parametrize("errors", [None, 5, "oops"], ids=["null", "number", "string"])
@pytest.mark.parametrize("path", ["tools", "text", "stream"])
def test_error_frame_with_malformed_errors_is_a_cli_error_and_tells_provider_auth(tmp_path, path, errors):
    # Stated change (plan review F3): `errors: null` used to hit a TypeError (-> claude_cli_parse on the blocking
    # branches, an unhandled TypeError on the stream path), so an auth text in the same frame never reached
    # provider_auth. No real frame on record has a malformed `errors`.
    from brain.bridge import provider_auth
    from brain.bridge.chat import StreamError

    persona_dir = _persona(tmp_path)
    stdout = "".join(_lines([_error_frame(errors=errors)]))
    if path == "stream":
        events = _stream(persona_dir, iter(stdout.splitlines(keepends=True)))
        assert isinstance(events[-1], StreamError) and events[-1].stage == "claude_cli_error"
    else:
        with pytest.raises(ProviderError) as excinfo:
            if path == "tools":
                _blocking(persona_dir, stdout)
            else:
                _text_branch(persona_dir, stdout)
        assert excinfo.value.stage == "claude_cli_error"
    assert provider_auth.state()["status"] == "expired"


@pytest.mark.parametrize("path", ["tools", "text"])
def test_exit_1_with_a_budget_sounding_error_but_another_subtype_is_still_an_exit_failure(tmp_path, path):
    # The exit != 0 carve-out is keyed on the subtype alone: an exit-1 frame that merely MENTIONS a budget
    # (an account cap, say) is not swallowed into the graceful reply.
    frame = _error_frame(subtype="error_during_execution", result="", errors=["Reached maximum budget ($5.00)"])
    stdout = "".join(_lines([frame]))
    with pytest.raises(ProviderError) as excinfo:
        if path == "tools":
            _blocking(_persona(tmp_path), stdout, returncode=1)
        else:
            _text_branch(_persona(tmp_path), stdout, returncode=1)
    assert excinfo.value.stage == "claude_cli_exit"


def test_streaming_path_with_the_real_over_budget_frame_is_graceful_and_logs_once(tmp_path):
    # The streaming path was always right (it reads the result frame before the exit code); this replays the
    # REAL frame at the REAL exit code through it, where the older budget tests use a hand-written frame at exit 0.
    from brain.bridge.chat import StreamDone

    persona_dir = _persona(tmp_path)
    events = _stream(persona_dir, iter(_stdout().splitlines(keepends=True)), exit_code=1)

    assert isinstance(events[-1], StreamDone)
    assert events[-1].content == _BUDGET_EXCEEDED_MSG
    assert len(_usage_rows(persona_dir)) == 1


def test_run_tool_loop_keeps_the_invocations_of_an_over_budget_turn(tmp_path):
    import subprocess
    from unittest.mock import patch

    from brain.bridge.chat import ChatMessage
    from brain.bridge.provider import ClaudeCliProvider
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = _persona(tmp_path)
    write_mcp_row = _mcp_row_during_run(persona_dir)

    def _run(cmd, **kwargs):
        write_mcp_row(cmd)  # what the brain-tools MCP child does mid-run
        return subprocess.CompletedProcess(cmd, 1, _stdout(), "")

    store, hebbian = MemoryStore(db_path=":memory:"), HebbianMatrix(db_path=":memory:")
    try:
        with patch("brain.bridge.provider.subprocess.run", side_effect=_run):
            response, invocations = run_tool_loop(
                [ChatMessage(role="system", content="SYS"), ChatMessage(role="user", content="hi")],
                provider=ClaudeCliProvider(model="haiku", timeout_seconds=60),
                tools=build_tools_list(),
                store=store,
                hebbian=hebbian,
                persona_dir=persona_dir,
            )
    finally:
        store.close()
        hebbian.close()

    assert response.content == _BUDGET_EXCEEDED_MSG
    assert [i["name"] for i in invocations] == ["search_memories"]
