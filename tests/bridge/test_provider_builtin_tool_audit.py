"""#273: the Claude CLI's built-in tool calls reach the audit log, and only the audit log.

Built-ins (WebSearch, WebFetch, ToolSearch, ...) run inside the `claude`
subprocess, so the brain-tools MCP child that writes
`tool_invocations.log.jsonl` never sees them. `_run_chat_stream` now records
each built-in `tool_use`/`tool_result` pair and writes it after the stream
with `origin: "cli_builtin"`; `_read_audit_lines_since` skips those rows so
they never become the turn's `dispatched_invocations` (owner ruling:
audit-only).

The fixture is a real `claude` 2.1.284 `--output-format stream-json --verbose`
capture (one ToolSearch, one WebSearch, one WebFetch), with local hook frames
and account connector tool names removed.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse, StreamDone, TextDelta
from brain.bridge.provider import (
    _MAX_TURN_BUDGET_USD,
    ClaudeCliProvider,
    ProviderError,
    _read_audit_lines_since,
)
from brain.bridge.server import _StreamingProxy
from brain.tools import NELL_TOOL_NAMES

_FIXTURE = Path(__file__).parent / "fixtures" / "cli_2_1_284_web_tools.ndjson"
_WEBFETCH_ID = "toolu_01Ji5TQRyiJ4rj6Azin47156"
_TOOLS = [{"name": "search_memories"}]  # non-empty → the production MCP argv branch


def _frames() -> list[dict]:
    return [json.loads(line) for line in _FIXTURE.read_text(encoding="utf-8").splitlines()]


def _lines(frames: list[dict]) -> list[str]:
    return [json.dumps(f) + "\n" for f in frames]


def _fake_popen(stdout, exit_code: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stdin = MagicMock()
    proc.wait.return_value = exit_code
    proc.poll.return_value = exit_code
    proc.returncode = exit_code
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    return proc


def _persona(tmp_path: Path, mode: str = "redacted") -> Path:
    persona_dir = tmp_path / "personas" / "scratch"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"mcp_audit_log_level": mode}), encoding="utf-8"
    )
    return persona_dir


def _log(persona_dir: Path) -> Path:
    return persona_dir / "tool_invocations.log.jsonl"


def _rows(persona_dir: Path) -> list[dict]:
    path = _log(persona_dir)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _stream(persona_dir: Path | None, stdout, *, tools=_TOOLS, exit_code: int = 0) -> list:
    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    options = {"persona_dir": str(persona_dir)} if persona_dir is not None else None
    with patch(
        "brain.bridge.provider.subprocess.Popen",
        return_value=_fake_popen(stdout, exit_code),
    ):
        return list(
            provider.chat_stream(
                [ChatMessage(role="user", content="hi")], tools=tools, options=options
            )
        )


# ---------------------------------------------------------------------------
# C1 — each web call produces exactly one audit row, on both argv branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tools", [_TOOLS, None], ids=["tools", "tools-less"])
def test_web_search_and_fetch_each_logged_once(tmp_path, tools):
    persona_dir = _persona(tmp_path)
    _stream(persona_dir, iter(_lines(_frames())), tools=tools)

    rows = _rows(persona_dir)
    for name in ("WebSearch", "WebFetch"):
        matching = [r for r in rows if r["name"] == name]
        assert len(matching) == 1, rows
        assert matching[0]["origin"] == "cli_builtin"
        assert matching[0]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# C2 — audit-only: built-in rows never reach dispatched_invocations
# ---------------------------------------------------------------------------


def _via_proxy(persona_dir: Path, stdout, *, exit_code: int = 0) -> ChatResponse:
    """Drive the real `_StreamingProxy.chat` (the live WS path and the real
    audit reader) over the real provider."""
    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    loop = asyncio.new_event_loop()

    async def _go() -> ChatResponse:
        proxy = _StreamingProxy(provider, asyncio.Queue(), asyncio.get_event_loop())
        return await asyncio.to_thread(
            proxy.chat,
            [ChatMessage(role="user", content="hi")],
            tools=_TOOLS,
            options={"persona_dir": str(persona_dir)},
        )

    try:
        with patch(
            "brain.bridge.provider.subprocess.Popen",
            return_value=_fake_popen(stdout, exit_code),
        ):
            return loop.run_until_complete(_go())
    finally:
        loop.close()


def _with_mcp_row_midstream(persona_dir: Path):
    """The fixture's stdout, with the MCP child appending its own row mid-stream."""
    mcp_row = {"name": "search_memories", "arguments": {}, "result_summary": "[]"}
    for i, line in enumerate(_lines(_frames())):
        if i == 4:
            with _log(persona_dir).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(mcp_row) + "\n")
        yield line


def test_builtin_rows_never_reach_dispatched_invocations(tmp_path):
    persona_dir = _persona(tmp_path)
    result = _via_proxy(persona_dir, _with_mcp_row_midstream(persona_dir))

    # The built-in rows ARE in the file, so the reader's filter was exercised.
    assert {r["name"] for r in _rows(persona_dir) if r.get("origin") == "cli_builtin"} >= {
        "WebSearch",
        "WebFetch",
    }
    assert [inv["name"] for inv in result.dispatched_invocations] == ["search_memories"]


# ---------------------------------------------------------------------------
# C3 — nothing private leaks in the default redacted mode, for any built-in
# ---------------------------------------------------------------------------

_QUERY = "SQLite 3.47 release date"
_URL = "https://sqlite.org/changes.html"
_FETCH_PROMPT = "What is the first release listed on this page?"
_TASK_ID = "toolu_task_zq"
_TASK_INPUT = {"subject": "ZQ-SUBJ", "description": "ZQ-DESC"}


def _tool_use(tool_id: str, name: str, tool_input: dict) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}],
        },
    }


def _tool_result(tool_id: str, content, *, is_error: bool | None = None) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_id, "content": content}
    if is_error is not None:
        block["is_error"] = is_error
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _before_result(extra: list[dict]) -> list[dict]:
    """The fixture with `extra` frames spliced in just before the result frame."""
    frames = _frames()
    return frames[:-1] + extra + frames[-1:]


def _with_task_create() -> list[dict]:
    return _before_result(
        [
            _tool_use(_TASK_ID, "TaskCreate", _TASK_INPUT),
            _tool_result(_TASK_ID, "Task #1 created: ZQ-SUBJ"),
        ]
    )


def test_redacted_mode_hides_every_builtin_argument_and_result(tmp_path):
    persona_dir = _persona(tmp_path, "redacted")
    _stream(persona_dir, iter(_lines(_with_task_create())))

    rows = [r for r in _rows(persona_dir) if r.get("origin") == "cli_builtin"]
    by_name = {r["name"]: r for r in rows}
    assert set(by_name) == {"ToolSearch", "WebSearch", "WebFetch", "TaskCreate"}
    expected_keys = {
        "WebSearch": {"query"},
        "WebFetch": {"url", "prompt"},
        "TaskCreate": {"subject", "description"},
    }
    for name, keys in expected_keys.items():
        args = by_name[name]["arguments"]
        assert set(args) == keys, (name, args)
        assert all(v == "[REDACTED]" for v in args.values()), (name, args)

    text = "\n".join(
        json.dumps(json.loads(x), ensure_ascii=False)
        for x in _log(persona_dir).read_text(encoding="utf-8").splitlines()
    )
    for secret in (_QUERY, _URL, _FETCH_PROMPT, "ZQ-SUBJ", "ZQ-DESC"):
        assert secret not in text, secret


def test_full_mode_keeps_them(tmp_path):
    persona_dir = _persona(tmp_path, "full")
    _stream(persona_dir, iter(_lines(_with_task_create())))

    by_name = {r["name"]: r for r in _rows(persona_dir) if r.get("origin") == "cli_builtin"}
    assert by_name["WebSearch"]["arguments"] == {"query": _QUERY}
    assert by_name["WebFetch"]["arguments"]["url"] == _URL
    assert by_name["TaskCreate"]["arguments"] == _TASK_INPUT
    assert _QUERY in by_name["WebSearch"]["result_summary"]


# ---------------------------------------------------------------------------
# C4 — every built-in is logged; brain-tools calls are not double-logged
# ---------------------------------------------------------------------------


def test_every_builtin_logged_and_brain_tools_not_double_logged(tmp_path):
    persona_dir = _persona(tmp_path)
    frames = _before_result(
        [
            _tool_use("toolu_mcp", "mcp__brain-tools__search_memories", {"query": "q"}),
            _tool_result("toolu_mcp", "[]"),
        ]
    )
    _stream(persona_dir, iter(_lines(frames)))

    rows = _rows(persona_dir)
    assert sorted(r["name"] for r in rows) == ["ToolSearch", "WebFetch", "WebSearch"]
    tool_search = next(r for r in rows if r["name"] == "ToolSearch")
    assert tool_search["outcome"] == "ok"  # its tool_result content is a list of blocks


# ---------------------------------------------------------------------------
# C5 — unanswered and failed calls, and the proxy's error path
# ---------------------------------------------------------------------------


def _truncated_after_webfetch() -> list[dict]:
    frames = _frames()
    cut = next(
        i
        for i, f in enumerate(frames)
        if any(
            isinstance(b, dict) and b.get("id") == _WEBFETCH_ID
            for b in (f.get("message") or {}).get("content") or []
        )
    )
    return frames[: cut + 1]


def test_unanswered_tool_use_logged_as_error(tmp_path):
    persona_dir = _persona(tmp_path)
    _stream(persona_dir, iter(_lines(_truncated_after_webfetch())))

    by_name = {r["name"]: r for r in _rows(persona_dir)}
    assert by_name["ToolSearch"]["outcome"] == "ok"
    for name in ("WebSearch", "WebFetch"):  # neither result arrived before EOF
        assert by_name[name]["outcome"] == "error"
        assert by_name[name]["error"] == "no result before the stream ended"


def test_is_error_result_logged_as_error(tmp_path):
    persona_dir = _persona(tmp_path)
    frames = _before_result(
        [
            _tool_use("toolu_bad", "WebFetch", {"url": "https://example.invalid/"}),
            _tool_result("toolu_bad", "fetch failed", is_error=True),
        ]
    )
    _stream(persona_dir, iter(_lines(frames)))

    bad = [r for r in _rows(persona_dir) if r["name"] == "WebFetch" and r["outcome"] == "error"]
    assert len(bad) == 1
    assert bad[0]["error"] == "tool reported an error"


def test_rows_on_disk_before_provider_error_propagates(tmp_path):
    persona_dir = _persona(tmp_path)
    with pytest.raises(ProviderError):
        _via_proxy(persona_dir, iter(_lines(_truncated_after_webfetch())), exit_code=1)
    names = {r["name"] for r in _rows(persona_dir)}
    assert {"ToolSearch", "WebSearch", "WebFetch"} <= names


# ---------------------------------------------------------------------------
# C6 — auditing can never change or break the turn
# ---------------------------------------------------------------------------


def _run_direct(persona_dir: Path | None, frames: list[dict]) -> list:
    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    with patch(
        "brain.bridge.provider.subprocess.Popen",
        return_value=_fake_popen(iter(_lines(frames))),
    ):
        return list(
            provider._run_chat_stream(
                cmd=["claude"], flat_prompt="hi", system_prompt=None, persona_dir=persona_dir
            )
        )


def test_audit_failure_never_changes_stream_events(tmp_path):
    frames = _frames()
    clean = _run_direct(_persona(tmp_path / "a"), frames)
    no_persona = _run_direct(None, frames)

    calls: list[str] = []

    def _boom(*args, **kwargs):
        calls.append(kwargs.get("name", "?"))
        raise RuntimeError("disk on fire")

    with patch("brain.mcp_server.audit.log_invocation", _boom):
        raising = _run_direct(_persona(tmp_path / "b"), frames)

    assert calls, "the raising stub was never called — this leg proved nothing"
    assert clean == no_persona == raising
    assert isinstance(clean[-1], StreamDone)


def test_malformed_frames_do_not_raise(tmp_path):
    persona_dir = _persona(tmp_path)
    junk = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    "not a block",
                    {"type": "tool_use"},
                    {"type": "tool_use", "id": 5, "name": "X"},
                    {"type": "tool_use", "id": "no-name", "name": None},
                    {"type": "tool_use", "id": "weird", "name": "Weird", "input": "str-input"},
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    "not a block",
                    {"type": "tool_result", "tool_use_id": "unknown"},
                    {"type": "tool_result", "tool_use_id": ["weird"]},  # unhashable
                    {"type": "tool_result", "tool_use_id": {"id": "weird"}},
                    {"type": "tool_result", "tool_use_id": "weird", "content": 5},
                ]
            },
        },
        {"type": "user", "message": {"content": "a bare string"}},
    ]
    frames = _before_result(junk)
    events = _stream(persona_dir, iter(_lines(frames)))

    # Identical to the same stream with auditing off (no persona dir).
    assert events == _run_direct(None, frames)
    assert isinstance(events[-1], StreamDone)
    weird = [r for r in _rows(persona_dir) if r["name"] == "Weird"]
    assert len(weird) == 1 and weird[0]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# C7 — two writers, append-only: neither row is lost
# ---------------------------------------------------------------------------


def _assert_mcp_then_builtins(log_text: str) -> None:
    rows = [json.loads(x) for x in log_text.splitlines() if x.strip()]  # every line whole
    assert rows and rows[0]["name"] == "search_memories", "the MCP row was lost or moved"
    assert rows[0].get("origin") is None
    assert {r["name"] for r in rows[1:]} == {"ToolSearch", "WebSearch", "WebFetch"}
    assert all(r.get("origin") == "cli_builtin" for r in rows[1:])


def test_mcp_row_then_builtin_flush_both_survive(tmp_path):
    persona_dir = _persona(tmp_path)
    _stream(persona_dir, _with_mcp_row_midstream(persona_dir))

    log_text = _log(persona_dir).read_text(encoding="utf-8")
    _assert_mcp_then_builtins(log_text)
    assert [r["name"] for r in _read_audit_lines_since(_log(persona_dir), 0)] == ["search_memories"]

    # The helper can fail: a log whose MCP line was clobbered by a truncating write.
    clobbered = "\n".join(log_text.splitlines()[1:])
    with pytest.raises(AssertionError):
        _assert_mcp_then_builtins(clobbered)


# ---------------------------------------------------------------------------
# C9 — argv and stdin are untouched (cost/cache can't move)
# ---------------------------------------------------------------------------

# Captured from main with the same inputs (re-captured after #332 added --tools=);
# tempfile paths normalised.
_EXPECTED_ARGV = [
    "claude",
    "-p",
    "--dangerously-skip-permissions",
    "--output-format",
    "stream-json",
    "--verbose",
    "--model",
    "haiku",
    "--max-budget-usd",
    "<BUDGET>",
    "--mcp-config",
    "<TMP>",
    "--allowedTools",
    "<ALLOWED>",
    "--tools=WebSearch,WebFetch",  # #332: the exclusive built-in allowlist
    "--disallowedTools",
    "Bash",
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Task",
    "TodoWrite",
    "NotebookEdit",
    "BashOutput",
    "KillShell",
    "--strict-mcp-config",
    "--system-prompt-file",
    "<TMP>",
]
_EXPECTED_STDIN = "hi\n\nVS"


def _capture_argv_and_stdin(tmp_path, monkeypatch) -> tuple[list[str], str]:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    persona_dir = _persona(tmp_path)
    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    proc = _fake_popen(iter(_lines(_frames()[-1:])))
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc) as popen:
        list(
            provider.chat_stream(
                [
                    ChatMessage(role="system", content="SYSTEM"),
                    ChatMessage(role="user", content="hi"),
                ],
                tools=_TOOLS,
                options={
                    "persona_dir": str(persona_dir),
                    "include_block_clock": False,
                    "volatile_suffix": "VS",
                },
            )
        )
    argv = list(popen.call_args.args[0])
    for flag in ("--mcp-config", "--system-prompt-file"):
        argv[argv.index(flag) + 1] = "<TMP>"
    # Exact values, computed from their sources rather than frozen, so adding a
    # brain tool (or retuning the budget) doesn't break this pin but any change
    # to how argv is built does.
    budget = argv.index("--max-budget-usd") + 1
    assert argv[budget] == str(_MAX_TURN_BUDGET_USD("haiku"))
    argv[budget] = "<BUDGET>"
    allowed = argv.index("--allowedTools")
    # The variadic list runs to the next flag, whatever that flag is.
    end = next(i for i in range(allowed + 1, len(argv)) if argv[i].startswith("--"))
    assert argv[allowed + 1 : end] == [f"mcp__brain-tools__{n}" for n in NELL_TOOL_NAMES]
    argv[allowed + 1 : end] = ["<ALLOWED>"]
    stdin = "".join(c.args[0] for c in proc.stdin.write.call_args_list)
    return argv, stdin


def test_argv_and_stdin_unchanged(tmp_path, monkeypatch):
    argv, stdin = _capture_argv_and_stdin(tmp_path, monkeypatch)
    assert argv == _EXPECTED_ARGV
    assert stdin == _EXPECTED_STDIN


def test_text_deltas_unaffected(tmp_path):
    """Sanity: the fixture has no stream_event deltas, so the reply arrives on
    StreamDone — auditing adds no event of its own."""
    events = _stream(_persona(tmp_path), iter(_lines(_frames())))
    assert not [e for e in events if isinstance(e, TextDelta)]
    assert [type(e).__name__ for e in events] == ["StreamDone"]


# ---------------------------------------------------------------------------
# #330 — the blocking tools path (`chat(tools=...)` → `_chat_with_mcp_tools`)
# ---------------------------------------------------------------------------


def _blocking(
    persona_dir: Path,
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
    during=None,
):
    """Drive the real `ClaudeCliProvider.chat(tools=...)` with `subprocess.run`
    stubbed. Returns (response, run_mock); callers assert the mock was called.
    `during(cmd)` runs inside the stub, i.e. while the "subprocess" is alive."""
    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    completed = subprocess.CompletedProcess(["claude"], returncode, stdout, stderr)

    def _run(cmd, **kwargs):
        if during is not None:
            during(cmd)
        return completed

    with patch("brain.bridge.provider.subprocess.run", side_effect=_run) as run:
        resp = provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=_TOOLS,
            options={"persona_dir": str(persona_dir)},
        )
    return resp, run


def _ndjson(frames: list[dict]) -> str:
    return "".join(_lines(frames))


def test_blocking_path_logs_web_search_and_fetch_once(tmp_path):
    persona_dir = _persona(tmp_path)
    _, run = _blocking(persona_dir, _ndjson(_frames()))

    assert run.called
    rows = _rows(persona_dir)
    for name in ("WebSearch", "WebFetch"):
        matching = [r for r in rows if r["name"] == name]
        assert len(matching) == 1, rows
        assert matching[0]["origin"] == "cli_builtin"
        assert matching[0]["outcome"] == "ok"


def _mcp_row_during_run(persona_dir: Path):
    """What the real MCP child does mid-run: append a brain-tools row stamped with
    the request_id it got through the `--mcp-config` env."""

    def _during(cmd):
        cfg = json.loads(Path(cmd[cmd.index("--mcp-config") + 1]).read_text(encoding="utf-8"))
        env = cfg["mcpServers"]["brain-tools"]["env"]
        row = {
            "name": "search_memories",
            "arguments": {},
            "result_summary": "{}",
            "outcome": "ok",
            "request_id": env["NELL_MCP_AUDIT_REQUEST_ID"],
        }
        with _log(persona_dir).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    return _during


def test_blocking_path_builtin_rows_stay_out_of_dispatched_invocations(tmp_path):
    persona_dir = _persona(tmp_path)
    resp, run = _blocking(
        persona_dir, _ndjson(_frames()), during=_mcp_row_during_run(persona_dir)
    )

    assert run.called
    # The built-in rows are on disk (so the filter was actually exercised) ...
    assert {r["name"] for r in _rows(persona_dir) if r.get("origin") == "cli_builtin"} >= {
        "WebSearch",
        "WebFetch",
    }
    # ... and only the brain-tools row is the turn's dispatched invocation.
    assert [d["name"] for d in resp.dispatched_invocations] == ["search_memories"]


def test_blocking_path_redacted_mode_hides_every_builtin_argument_and_result(tmp_path):
    persona_dir = _persona(tmp_path, "redacted")
    _, run = _blocking(persona_dir, _ndjson(_with_task_create()))

    assert run.called
    by_name = {r["name"]: r for r in _rows(persona_dir) if r.get("origin") == "cli_builtin"}
    assert set(by_name) == {"ToolSearch", "WebSearch", "WebFetch", "TaskCreate"}
    expected_keys = {
        "WebSearch": {"query"},
        "WebFetch": {"url", "prompt"},
        "TaskCreate": {"subject", "description"},
    }
    for name, keys in expected_keys.items():
        args = by_name[name]["arguments"]
        assert set(args) == keys, (name, args)
        assert all(v == "[REDACTED]" for v in args.values()), (name, args)

    text = _log(persona_dir).read_text(encoding="utf-8")
    for secret in (_QUERY, _URL, _FETCH_PROMPT, "ZQ-SUBJ", "ZQ-DESC"):
        assert secret not in text, secret


def test_blocking_path_full_mode_keeps_builtin_arguments(tmp_path):
    persona_dir = _persona(tmp_path, "full")
    _, run = _blocking(persona_dir, _ndjson(_with_task_create()))

    assert run.called
    by_name = {r["name"]: r for r in _rows(persona_dir) if r.get("origin") == "cli_builtin"}
    assert by_name["WebSearch"]["arguments"] == {"query": _QUERY}
    assert by_name["WebFetch"]["arguments"]["url"] == _URL
    assert by_name["TaskCreate"]["arguments"] == _TASK_INPUT


def test_blocking_path_logs_every_builtin_but_not_brain_tools(tmp_path):
    persona_dir = _persona(tmp_path)
    frames = _before_result(
        [
            _tool_use("toolu_mcp", "mcp__brain-tools__search_memories", {"query": "q"}),
            _tool_result("toolu_mcp", "[]"),
        ]
    )
    _, run = _blocking(persona_dir, _ndjson(frames))

    assert run.called
    rows = _rows(persona_dir)
    assert sorted(r["name"] for r in rows) == ["ToolSearch", "WebFetch", "WebSearch"]
    tool_search = next(r for r in rows if r["name"] == "ToolSearch")
    assert tool_search["outcome"] == "ok"  # list-shaped tool_result content


# C4 — everything downstream of the result frame is unchanged


def test_blocking_path_returns_result_text_and_logs_one_usage_row(tmp_path):
    persona_dir = _persona(tmp_path)
    result = _frames()[-1]
    resp, run = _blocking(persona_dir, _ndjson(_frames()))

    assert run.called
    assert resp.content == result["result"]
    rows = [
        json.loads(x)
        for x in (persona_dir / "chat_usage.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    # Pinned to fields only the result frame carries (session_id is on every frame).
    assert rows[0]["call_type"] == "chat"
    assert rows[0]["total_cost_usd"] == result["total_cost_usd"]
    assert rows[0]["cache_read_input_tokens"] == result["usage"]["cache_read_input_tokens"]
    assert rows[0]["output_tokens"] == result["usage"]["output_tokens"]


def _with_result(**overrides) -> str:
    frames = _frames()
    frames[-1] = {**frames[-1], **overrides}
    return _ndjson(frames)


def test_blocking_path_budget_exceeded_result_gives_the_budget_message(tmp_path):
    from brain.bridge.provider import _BUDGET_EXCEEDED_MSG

    persona_dir = _persona(tmp_path)
    stdout = _with_result(
        subtype="error_max_budget_usd", is_error=True, result="partial answer", errors=[]
    )
    resp, run = _blocking(persona_dir, stdout)

    assert run.called
    assert resp.content == f"partial answer\n\n{_BUDGET_EXCEEDED_MSG}"


def test_blocking_path_is_error_result_raises_claude_cli_error(tmp_path):
    persona_dir = _persona(tmp_path)
    stdout = _with_result(is_error=True, result="API Error: 529 overloaded", subtype="success")
    with pytest.raises(ProviderError) as excinfo:
        _blocking(persona_dir, stdout)

    assert excinfo.value.stage == "claude_cli_error"
    assert "529" in str(excinfo.value)


def test_blocking_path_accepts_a_bare_single_json_result_object(tmp_path):
    persona_dir = _persona(tmp_path)
    resp, run = _blocking(persona_dir, json.dumps({"result": "hello back"}))

    assert run.called
    assert resp.content == "hello back"


def test_blocking_path_lone_typed_non_result_frame_is_a_parse_error_not_a_recovery(tmp_path):
    from brain.bridge import provider_auth

    persona_dir = _persona(tmp_path)
    provider_auth.note_cli_failure("Invalid API key · OAuth session expired")
    assert provider_auth.state()["status"] == "expired"

    init_only = json.dumps(_frames()[0])
    with pytest.raises(ProviderError) as excinfo:
        _blocking(persona_dir, init_only)

    assert excinfo.value.stage == "claude_cli_parse"
    assert provider_auth.state()["status"] == "expired"  # not cleared by a non-result frame


# C5 — failure shapes keep their error codes AND their detail text

_AUTH_TEXT = "Invalid API key · OAuth session expired"


@pytest.mark.parametrize("separator", ["", "\u2028", "\u0085"], ids=["plain", "u2028", "u0085"])
def test_blocking_path_nonzero_exit_ndjson_keeps_structured_detail_and_auth_state(
    tmp_path, separator
):
    from brain.bridge import provider as provider_mod
    from brain.bridge import provider_auth

    persona_dir = _persona(tmp_path)
    text = f"{_AUTH_TEXT}{separator} please run /login"
    failure = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": 401,
        "result": text,
    }
    # ensure_ascii=False so a raw separator reaches the parser (the _lines helper escapes it).
    stdout = "\n".join(
        json.dumps(f, ensure_ascii=False) for f in (_frames()[0], failure)
    ) + "\n"

    # Control: the pre-change plain-JSON stdout for the same failure.
    control = provider_mod._claude_failure_detail_text(
        subprocess.CompletedProcess(["claude"], 1, json.dumps(failure, ensure_ascii=False), "")
    )
    assert "api_error_status=401" in control and _AUTH_TEXT in control

    with pytest.raises(ProviderError) as excinfo:
        _blocking(persona_dir, stdout, returncode=1)

    assert excinfo.value.stage == "claude_cli_exit"
    assert str(excinfo.value).endswith(f"exit 1: {control}"), str(excinfo.value)
    assert provider_auth.state()["status"] == "expired"


@pytest.mark.parametrize(
    "stdout",
    [_ndjson(_frames()[:-1]), "Error: not json at all\n", ""],
    ids=["ndjson-without-result", "plain-text", "empty"],
)
def test_blocking_path_without_a_result_frame_is_claude_cli_parse(tmp_path, stdout):
    persona_dir = _persona(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        _blocking(persona_dir, stdout)

    assert excinfo.value.stage == "claude_cli_parse"


def test_blocking_path_nonzero_exit_still_flushes_builtin_calls_seen(tmp_path):
    persona_dir = _persona(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        _blocking(persona_dir, _ndjson(_truncated_after_webfetch()), returncode=1)

    assert excinfo.value.stage == "claude_cli_exit"
    by_name = {r["name"]: r for r in _rows(persona_dir)}
    assert by_name["ToolSearch"]["outcome"] == "ok"
    for name in ("WebSearch", "WebFetch"):  # neither result arrived before the exit
        assert by_name[name]["outcome"] == "error"
        assert by_name[name]["error"] == "no result before the stream ended"


def test_blocking_path_is_error_tool_result_logged_as_error(tmp_path):
    persona_dir = _persona(tmp_path)
    frames = _before_result(
        [
            _tool_use("toolu_bad", "WebFetch", {"url": "https://example.invalid/"}),
            _tool_result("toolu_bad", "fetch failed", is_error=True),
        ]
    )
    _, run = _blocking(persona_dir, _ndjson(frames))

    assert run.called
    bad = [r for r in _rows(persona_dir) if r["name"] == "WebFetch" and r["outcome"] == "error"]
    assert len(bad) == 1
    assert bad[0]["error"] == "tool reported an error"


# C6 — auditing can never change or break the turn

_JUNK_LINES = [
    "not json at all",
    "[1, 2, 3]",
    "null",
    json.dumps({"type": "assistant", "message": "a string, not a dict"}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebFetch"}]}}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": 7, "name": "WebFetch"}]}}),
    json.dumps({"type": "user", "message": {"content": 5}}),
    json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": None}]}}),
    json.dumps({"type": "user"}),
]


def test_blocking_path_malformed_frames_leave_the_response_unchanged(tmp_path):
    clean_dir = _persona(tmp_path / "clean")
    clean, _ = _blocking(clean_dir, _ndjson(_frames()))

    noisy_dir = _persona(tmp_path / "noisy")
    frames = _lines(_frames())
    noisy_stdout = "".join(frames[:-1]) + "\n".join(_JUNK_LINES) + "\n" + frames[-1]
    noisy, run = _blocking(noisy_dir, noisy_stdout)

    assert run.called
    assert (noisy.content, noisy.tool_calls, noisy.dispatched_invocations) == (
        clean.content,
        clean.tool_calls,
        clean.dispatched_invocations,
    )
    assert [r["name"] for r in _rows(noisy_dir)] == [r["name"] for r in _rows(clean_dir)]


def test_blocking_path_raw_unicode_line_separators_do_not_split_frames(tmp_path):
    persona_dir = _persona(tmp_path)  # redacted: the log itself carries no raw separator
    sep_text = "page a\u2028b\u2029c\u0085d end"
    frames = _before_result(
        [
            _tool_use("toolu_sep", "WebFetch", {"url": "https://example.invalid/sep"}),
            _tool_result("toolu_sep", sep_text),
        ]
    )
    frames[-1] = {**frames[-1], "result": f"answer {sep_text}"}
    # ensure_ascii=False: the existing `_lines` helper escapes the separators, which
    # would make this test vacuous.
    stdout = "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in frames)
    assert "\u2028" in stdout and len(stdout.splitlines()) > len(stdout.split("\n")) - 1

    resp, run = _blocking(persona_dir, stdout)

    assert run.called
    assert resp.content == f"answer {sep_text}"
    fetch_rows = [r for r in _rows(persona_dir) if r["name"] == "WebFetch"]
    assert len(fetch_rows) == 2  # the fixture's fetch plus this one
    assert all(r["outcome"] == "ok" for r in fetch_rows)  # the separator frame was not dropped


def test_blocking_path_audit_write_failure_does_not_change_the_response(tmp_path):
    clean_dir = _persona(tmp_path / "clean")
    clean, _ = _blocking(clean_dir, _ndjson(_frames()))

    broken_dir = _persona(tmp_path / "broken")
    boom = MagicMock(side_effect=RuntimeError("disk on fire"))
    with patch("brain.mcp_server.audit.log_invocation", boom):
        broken, run = _blocking(broken_dir, _ndjson(_frames()))

    assert run.called
    assert boom.called  # the failure path was really exercised
    assert (broken.content, broken.tool_calls, broken.dispatched_invocations) == (
        clean.content,
        clean.tool_calls,
        clean.dispatched_invocations,
    )


def _plain_json_failure() -> str:
    return json.dumps(
        {"type": "result", "is_error": True, "api_error_status": 401, "result": _AUTH_TEXT}
    )


def test_generate_plain_json_failure_detail_and_auth_state_unchanged():
    from brain.bridge import provider_auth

    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    failed = subprocess.CompletedProcess(["claude"], 1, _plain_json_failure(), "")
    with patch("brain.bridge.provider.subprocess.run", return_value=failed) as run:
        with pytest.raises(RuntimeError) as excinfo:  # generate() raises RuntimeError
            provider.generate("hi")

    assert run.called
    assert str(excinfo.value).endswith(f"(exit 1): api_error_status=401; is_error=True; {_AUTH_TEXT}")
    assert provider_auth.state()["status"] == "expired"


def test_tools_less_chat_plain_json_failure_detail_and_auth_state_unchanged(tmp_path):
    from brain.bridge import provider_auth

    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    failed = subprocess.CompletedProcess(["claude"], 1, _plain_json_failure(), "")
    with patch("brain.bridge.provider.subprocess.run", return_value=failed) as run:
        with pytest.raises(ProviderError) as excinfo:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=None,
                options={"persona_dir": str(_persona(tmp_path))},
            )

    assert run.called
    assert str(excinfo.value).endswith(f"exit 1: api_error_status=401; is_error=True; {_AUTH_TEXT}")
    assert provider_auth.state()["status"] == "expired"


# C7 — no cost/cache surface change: the argv differs from origin/main's in exactly
# the output-format pair; stdin and the system-prompt file are byte-identical.

# Captured from `origin/main` (808acff5) on 2026-10-10 by running the blocking tools
# path with a stubbed `subprocess.run` (tempfile paths normalised to <TMPFILE>).
# `--allowedTools` names and the budget are derived, so adding a tool or retuning the
# budget does not break this test; every other token is pinned.
def _old_main_argv() -> list[str]:
    return [
        "claude", "-p", "--dangerously-skip-permissions",
        "--output-format", "json",
        "--model", "haiku",
        "--mcp-config", "<TMPFILE>",
        "--allowedTools", *[f"mcp__brain-tools__{n}" for n in NELL_TOOL_NAMES],
        "--tools=WebSearch,WebFetch",
        "--disallowedTools", "Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task",
        "TodoWrite", "NotebookEdit", "BashOutput", "KillShell",
        "--strict-mcp-config",
        "--max-budget-usd", str(_MAX_TURN_BUDGET_USD("haiku")),
        "--system-prompt-file", "<TMPFILE>",
    ]  # fmt: skip


def test_blocking_path_argv_differs_from_origin_main_only_in_the_output_format(tmp_path):
    persona_dir = _persona(tmp_path)
    seen: dict = {}

    def _during(cmd):
        seen["cmd"] = list(cmd)
        seen["system"] = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")

    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    completed = subprocess.CompletedProcess(["claude"], 0, _ndjson(_frames()), "")

    def _run(cmd, **kwargs):
        _during(cmd)
        seen["input"] = kwargs.get("input")
        return completed

    with patch("brain.bridge.provider.subprocess.run", side_effect=_run) as run:
        provider.chat(
            [
                ChatMessage(role="system", content="SYSTEM PROMPT"),
                ChatMessage(role="user", content="hello there"),
            ],
            tools=_TOOLS,
            options={
                "persona_dir": str(persona_dir),
                "include_block_clock": False,
                "volatile_suffix": "VOLATILE TAIL",
            },
        )

    assert run.called
    normalised = [
        "<TMPFILE>" if c.startswith("/") and c.endswith((".json", ".txt", ".md", ".tmp", ".prompt")) or c.startswith(tempfile.gettempdir()) else c
        for c in seen["cmd"]
    ]
    expected = _old_main_argv()
    i = expected.index("json")
    expected[i : i + 1] = ["stream-json", "--verbose"]
    assert normalised == expected
    assert seen["input"] == "hello there\n\nVOLATILE TAIL"
    assert seen["system"] == "SYSTEM PROMPT"


# The stated RAT2 extension: any stdout whose whole-string parse fails and which holds a
# `{"type":"result"}` line now gets the structured detail from the two plain-`json` callers.
_SHAPES_WITH_A_RESULT_LINE = {
    "leading-garbage-line": "Warning: something noisy\n" + _plain_json_failure() + "\n",
    "trailing-warning": _plain_json_failure() + "\nWarning: something noisy\n",
}


@pytest.mark.parametrize("stdout", list(_SHAPES_WITH_A_RESULT_LINE.values()), ids=list(_SHAPES_WITH_A_RESULT_LINE))
def test_generate_failure_with_a_result_line_among_other_text_gets_structured_detail(stdout):
    from brain.bridge import provider_auth

    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    failed = subprocess.CompletedProcess(["claude"], 1, stdout, "")
    with patch("brain.bridge.provider.subprocess.run", return_value=failed) as run:
        with pytest.raises(RuntimeError) as excinfo:
            provider.generate("hi")

    assert run.called
    assert str(excinfo.value).endswith(f"(exit 1): api_error_status=401; is_error=True; {_AUTH_TEXT}")
    assert provider_auth.state()["status"] == "expired"


@pytest.mark.parametrize("stdout", list(_SHAPES_WITH_A_RESULT_LINE.values()), ids=list(_SHAPES_WITH_A_RESULT_LINE))
def test_tools_less_chat_failure_with_a_result_line_among_other_text_gets_structured_detail(
    tmp_path, stdout
):
    from brain.bridge import provider_auth

    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    failed = subprocess.CompletedProcess(["claude"], 1, stdout, "")
    with patch("brain.bridge.provider.subprocess.run", return_value=failed) as run:
        with pytest.raises(ProviderError) as excinfo:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=None,
                options={"persona_dir": str(_persona(tmp_path))},
            )

    assert run.called
    assert str(excinfo.value).endswith(f"exit 1: api_error_status=401; is_error=True; {_AUTH_TEXT}")
    assert provider_auth.state()["status"] == "expired"


def test_plain_text_failure_stdout_keeps_its_old_detail_on_both_plain_json_callers(tmp_path):
    provider = ClaudeCliProvider(model="haiku", timeout_seconds=60)
    failed = subprocess.CompletedProcess(["claude"], 1, "segfault: something broke\n", "")
    with patch("brain.bridge.provider.subprocess.run", return_value=failed) as run:
        with pytest.raises(RuntimeError) as generate_exc:
            provider.generate("hi")
        with pytest.raises(ProviderError) as chat_exc:
            provider.chat(
                [ChatMessage(role="user", content="hi")],
                tools=None,
                options={"persona_dir": str(_persona(tmp_path))},
            )

    assert run.call_count == 2
    assert str(generate_exc.value).endswith("(exit 1): stdout=segfault: something broke")
    assert str(chat_exc.value).endswith("exit 1: stdout=segfault: something broke")


def test_blocking_path_accepts_a_pretty_printed_typed_result_object(tmp_path):
    persona_dir = _persona(tmp_path)
    pretty = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "spread out"}, indent=2)
    assert len(pretty.splitlines()) > 1

    resp, run = _blocking(persona_dir, pretty)

    assert run.called
    assert resp.content == "spread out"


def test_blocking_path_still_truncates_a_role_leak_in_the_reply(tmp_path):
    persona_dir = _persona(tmp_path)
    leaky = "the real reply\n\nUser: a fabricated next turn"
    resp, run = _blocking(persona_dir, _with_result(result=leaky))

    assert run.called
    assert resp.content == "the real reply"
