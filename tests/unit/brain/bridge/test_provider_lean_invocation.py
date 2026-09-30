"""Task A1 — lean CLI invocation.

Guards that every ClaudeCliProvider argv builder:
1. Adds ``--disallowedTools`` covering the built-in tools she has no business
   calling (trims their definition tokens from cache-creation cost). This is
   POLICY, not dead weight — anything off the list is genuinely callable, so
   the list decides what she can reach (#71).
2. Adds ``--strict-mcp-config`` to pin the session to the configured MCP server.

Since #329 all four ``claude -p`` spawn sites apply the posture through
``_apply_lean_flags``: the CLI's exclusive ``--tools=`` allowlist (WebSearch,WebFetch
for chat; nothing for background ``generate()``) when the CLI supports it, then the
disallow list and ``--strict-mcp-config``.
"""

from __future__ import annotations

import pytest

from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED


def test_builtin_disallow_list_covers_the_costly_tools():
    for t in ("Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task"):
        assert t in _BUILTIN_TOOLS_DISALLOWED, f"{t!r} missing from _BUILTIN_TOOLS_DISALLOWED"


def test_web_tools_stay_callable_at_chat_time():
    """WebFetch/WebSearch must NOT be disallowed — they are a real capability,
    not dead weight (issue #71).

    History: 87bfc692 swept them in with the dev tools on the rationale that
    the persona "can't call them anyway — she's restricted to
    mcp__brain-tools__*". That was false: --allowedTools is a PERMISSION list,
    not an exclusive one, and --dangerously-skip-permissions is on every call —
    so the built-ins were genuinely reachable. (Since #329 the exclusive --tools=
    allowlist names WebSearch,WebFetch for chat, so they must stay off the
    disallow list too, or chat would lose them on a CLI without --tools.) Blocking Bash/Edit/Write/Task is deliberate policy;
    blocking web access was collateral. This canary stops the next lean pass
    from sweeping them back in.
    """
    for t in ("WebFetch", "WebSearch"):
        assert t not in _BUILTIN_TOOLS_DISALLOWED, (
            f"{t!r} is disallowed — chat-time web access is gone again (#71)"
        )


def test_apply_lean_flags_adds_disallowed_and_strict():
    from brain.bridge.provider import _BUILTIN_TOOLS_DISALLOWED, _apply_lean_flags
    cmd: list[str] = []
    _apply_lean_flags(cmd)
    assert "--disallowedTools" in cmd
    assert "--strict-mcp-config" in cmd
    for t in _BUILTIN_TOOLS_DISALLOWED:
        assert t in cmd, f"{t!r} not forwarded to cmd by _apply_lean_flags"


# ---------------------------------------------------------------------------
# #329 — an exclusive built-in allowlist (`--tools=`) on every claude spawn.
# New names are imported inside each test (not at module level) so this file
# still collects against the pre-change provider.py (the C11 swap-back check).
# ---------------------------------------------------------------------------

_RESULT_JSON = '{"type": "result", "subtype": "success", "is_error": false, "result": "ok"}'
_CHAT_TOOLS = "--tools=WebSearch,WebFetch"


def _fake_popen(stdout_lines: list[str], exit_code: int = 0):
    from unittest.mock import MagicMock

    proc = MagicMock()
    proc.stdout = iter(stdout_lines)
    proc.stdin = MagicMock()
    proc.wait.return_value = exit_code
    proc.poll.return_value = exit_code
    proc.returncode = exit_code
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    return proc


def _capture_argv(site: str, tmp_path) -> list[str]:
    """Drive one real ClaudeCliProvider entry point with the spawn faked and
    return the argv it built."""
    import subprocess
    from unittest.mock import patch

    from brain.bridge.chat import ChatMessage
    from brain.bridge.provider import ClaudeCliProvider

    provider = ClaudeCliProvider(model="haiku", timeout_seconds=30)
    msgs = [ChatMessage(role="user", content="hi")]
    persona = tmp_path / "persona"
    persona.mkdir(exist_ok=True)
    tools = [{"name": "search_memories"}]

    if site.startswith("chat_stream"):
        with patch(
            "brain.bridge.provider.subprocess.Popen",
            return_value=_fake_popen([_RESULT_JSON + "\n"]),
        ) as popen:
            list(
                provider.chat_stream(
                    msgs,
                    tools=tools if site == "chat_stream_tools" else None,
                    options={"persona_dir": str(persona)},
                )
            )
        return list(popen.call_args.args[0])

    done = subprocess.CompletedProcess(args=[], returncode=0, stdout=_RESULT_JSON, stderr="")
    with patch("brain.bridge.provider.subprocess.run", return_value=done) as run:
        if site == "generate":
            provider.generate("hi")
        elif site == "chat_text":
            provider.chat(msgs, tools=None)
        else:  # chat_mcp_tools
            provider.chat(msgs, tools=tools, options={"persona_dir": str(persona)})
    return list(run.call_args.args[0])


_SITES = ["generate", "chat_text", "chat_stream_tools", "chat_stream_no_tools", "chat_mcp_tools"]


@pytest.mark.parametrize("site", _SITES)
def test_every_site_applies_the_posture(site, tmp_path):
    """C1: background calls get no built-ins at all; every chat path gets
    exactly the web tools. The disallow list and --strict-mcp-config stay on
    as the second layer."""
    argv = _capture_argv(site, tmp_path)
    expected = "--tools=" if site == "generate" else _CHAT_TOOLS
    assert [a for a in argv if a.startswith("--tools")] == [expected], argv
    assert "--disallowedTools" in argv, argv
    assert "--strict-mcp-config" in argv, argv


def _count_model_spawn_sites(source: str) -> int:
    """Argv literals that start a model call: `"claude",` followed by `"-p",`."""
    import re

    return len(re.findall(r'"claude",\s*"-p",', source))


def test_every_claude_spawn_site_is_covered():
    """C2: a fifth `claude -p` argv added to provider.py fails here until the
    site-by-site posture test (test_every_site_applies_the_posture) covers it."""
    from pathlib import Path

    import brain.bridge.provider as provider_module

    source = Path(provider_module.__file__).read_text(encoding="utf-8")
    assert _count_model_spawn_sites(source) == len(_SITES) - 1, (
        "a claude -p spawn site was added or removed; cover it in _SITES "
        "(chat_stream has two branches, hence len(_SITES) - 1)"
    )
    # The counter can fail: one more literal is seen.
    assert _count_model_spawn_sites(source + '\ncmd = ["claude", "-p", "x"]\n') == len(_SITES)


_SENTINEL = "  --output-format <format>              Output format (only works with --print):\n"
_OPTION = "  --tools <tools...>                    Specify the list of available tools from\n"
# 2.1.284's --restricted description wraps onto a line that starts with --tools.
_RESTRICTED_WRAP = "                                        --tools names them, and ignores user,\n"
_INPUT_MISSING = (
    "Error: Input must be provided either through stdin or as a prompt argument when using --print"
)
_UNKNOWN_TOOLS = "error: unknown option '--tools='"


def _probe_fake(help_out: str, confirm_err: str = _INPUT_MISSING, calls: list | None = None):
    """A subprocess.run stand-in for the two probe commands: `claude --help`,
    and the parse-only confirmation `claude --tools= -p` (empty stdin)."""
    import subprocess

    def _run(argv, **kwargs):
        if calls is not None:
            calls.append(list(argv))
        if "--help" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=help_out, stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=confirm_err)

    return _run


@pytest.mark.parametrize(
    ("help_out", "confirm_err", "expected"),
    [
        (_SENTINEL + _OPTION, _INPUT_MISSING, True),  # listed
        (_SENTINEL, _UNKNOWN_TOOLS, False),  # (a) absent, confirmed unknown
        (_SENTINEL + _RESTRICTED_WRAP, _UNKNOWN_TOOLS, False),  # (c) prose mention only
        (_SENTINEL + "  --tools, --allowed-builtins <tools...>  x\n", _UNKNOWN_TOOLS, True),  # (e)
        (_SENTINEL + "  -t, --tools <tools...>  x\n", _UNKNOWN_TOOLS, True),
        (_SENTINEL + "  --builtins, --tools <tools...>  x\n", _UNKNOWN_TOOLS, True),
        (_SENTINEL + "  --tools [tools...]  x\n", _UNKNOWN_TOOLS, True),
        (_SENTINEL, _INPUT_MISSING, True),  # (f) hidden from --help but accepted
    ],
)
def test_probe_reads_support_definitively(monkeypatch, caplog, help_out, confirm_err, expected):
    """C6 (a)(c)(e)(f): a clean `--help` with the sentinel line is definitive.
    The option line decides; if it's absent, a parse-only `--tools=` run
    confirms before "unsupported" is cached. False logs one WARNING, and the
    argv then carries no --tools= but keeps the rest of the posture."""
    import logging

    from brain.bridge import provider

    monkeypatch.setattr(provider.subprocess, "run", _probe_fake(help_out, confirm_err))
    provider._reset_tools_flag_cache()
    with caplog.at_level(logging.WARNING, logger="brain.bridge.provider"):
        assert provider._cli_supports_tools_flag() is expected
        assert provider._cli_supports_tools_flag() is expected  # cached
        cmd: list[str] = []
        provider._apply_lean_flags(cmd, builtins=provider._CHAT_BUILTIN_TOOLS)
    assert (_CHAT_TOOLS in cmd) is expected
    assert "--disallowedTools" in cmd and "--strict-mcp-config" in cmd
    warnings = [r for r in caplog.records if "--tools" in r.getMessage()]
    assert len(warnings) == (0 if expected else 1)


def _raising(exc):
    def _run(argv, **kwargs):
        raise exc

    return _run


def _completed(rc: int, stdout: str):
    import subprocess

    def _run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr="")

    return _run


@pytest.mark.parametrize(
    "failing_run",
    [
        _raising(OSError("claude not on PATH")),
        _raising(__import__("subprocess").TimeoutExpired(["claude", "--help"], 10)),
        _completed(1, _SENTINEL + _OPTION),  # (d) ran but exited non-zero
        _completed(0, ""),  # empty output
        _completed(0, "some other text\n"),  # not the help format (no sentinel)
        _completed(-9, ""),  # killed
    ],
    ids=["oserror", "timeout", "rc1", "empty", "no-sentinel", "killed"],
)
def test_indeterminate_probe_is_not_cached_and_retries(monkeypatch, caplog, failing_run):
    """C6 (b)(d): a probe that couldn't tell falls back for this call, isn't
    cached, warns once per 60 s window, and a later probe can still say True."""
    import logging

    from brain.bridge import provider

    clock = [1000.0]
    monkeypatch.setattr(provider, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(provider.subprocess, "run", failing_run)
    provider._reset_tools_flag_cache()
    with caplog.at_level(logging.WARNING, logger="brain.bridge.provider"):
        assert provider._cli_supports_tools_flag() is False
        assert provider._TOOLS_FLAG_SUPPORTED is None  # not cached
        clock[0] += 30  # inside the window: no new probe, no new warning
        assert provider._cli_supports_tools_flag() is False
        assert len([r for r in caplog.records if "--tools" in r.getMessage()]) == 1

        clock[0] += 31  # window over; the CLI now answers properly
        monkeypatch.setattr(provider.subprocess, "run", _probe_fake(_SENTINEL + _OPTION))
        assert provider._cli_supports_tools_flag() is True
        assert provider._TOOLS_FLAG_SUPPORTED is True


def test_probe_runs_once_when_definitive(monkeypatch, tmp_path):
    """C7: five argv builds after a definitive answer spawn one probe."""
    from brain.bridge import provider

    calls: list = []
    monkeypatch.setattr(provider.subprocess, "run", _probe_fake(_SENTINEL + _OPTION, calls=calls))
    provider._reset_tools_flag_cache()
    for _ in range(5):
        provider._apply_lean_flags([], builtins=provider._CHAT_BUILTIN_TOOLS)
    assert calls == [["claude", "--help"]]


def test_probe_is_thread_safe(monkeypatch):
    """C7: 8 threads released together get the same True from exactly one
    probe; the rest wait on the lock instead of probing again."""
    import threading
    import time

    from brain.bridge import provider

    calls: list = []
    fake = _probe_fake(_SENTINEL + _OPTION, calls=calls)

    def _slow(argv, **kwargs):
        time.sleep(0.05)  # hold the lock long enough for every thread to queue
        return fake(argv, **kwargs)

    monkeypatch.setattr(provider.subprocess, "run", _slow)
    provider._reset_tools_flag_cache()
    start = threading.Barrier(8)
    results: list = []

    def _worker():
        start.wait()
        results.append(provider._cli_supports_tools_flag())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert results == [True] * 8
    assert calls == [["claude", "--help"]]


def _has_empty_element(argv: list[str]) -> bool:
    return any(a == "" for a in argv)


@pytest.mark.parametrize("supported", [True, False], ids=["tools-flag", "fallback"])
def test_no_empty_argv_elements(tmp_path, supported):
    """C8: no site ever passes a bare "" (Windows CreateProcess quoting could
    drop it and --tools would swallow the next flag). The empty allowlist is
    the single element "--tools=" instead."""
    from brain.bridge import provider

    provider._TOOLS_FLAG_SUPPORTED = supported
    for site in _SITES:
        argv = _capture_argv(site, tmp_path)
        assert not _has_empty_element(argv), (site, argv)
        if not supported:  # C6(a): every site drops --tools= but keeps the rest
            assert not [a for a in argv if a.startswith("--tools")], (site, argv)
            assert "--disallowedTools" in argv and "--strict-mcp-config" in argv, (site, argv)
    # The check can fail.
    assert _has_empty_element(["claude", "--tools", ""])


_REJECTION = "error: unknown option '--tools=WebSearch,WebFetch'"


def _failing_generate(stdout: str, stderr: str) -> None:
    import subprocess
    from unittest.mock import patch

    from brain.bridge.provider import ClaudeCliProvider

    done = subprocess.CompletedProcess(args=[], returncode=1, stdout=stdout, stderr=stderr)
    with patch("brain.bridge.provider.subprocess.run", return_value=done):
        with pytest.raises(RuntimeError, match="exit 1"):
            ClaudeCliProvider(model="haiku", timeout_seconds=30).generate("hi")


def _failing_stream(stdout_lines: list[str], stderr: str, tmp_path) -> None:
    from unittest.mock import patch

    from brain.bridge.chat import ChatMessage
    from brain.bridge.provider import ClaudeCliProvider

    proc = _fake_popen(stdout_lines, exit_code=1)
    proc.stderr.read.return_value = stderr
    with patch("brain.bridge.provider.subprocess.Popen", return_value=proc):
        list(
            ClaudeCliProvider(model="haiku", timeout_seconds=30).chat_stream(
                [ChatMessage(role="user", content="hi")],
                tools=None,
                options={"persona_dir": str(tmp_path)},
            )
        )


@pytest.mark.parametrize("path", ["run", "popen"])
def test_rejection_flips_to_fallback(caplog, tmp_path, path):
    """C14 (owner ruling): when the CLI rejects --tools before running anything
    (rc 1, empty stdout, the parser's message on stderr), this process falls back
    so the next calls work, with one WARNING."""
    import logging

    from brain.bridge import provider

    with caplog.at_level(logging.WARNING, logger="brain.bridge.provider"):
        if path == "run":
            _failing_generate("", _REJECTION)
        else:
            _failing_stream([], _REJECTION, tmp_path)
    assert provider._TOOLS_FLAG_SUPPORTED is False
    cmd: list[str] = []
    provider._apply_lean_flags(cmd, builtins=provider._CHAT_BUILTIN_TOOLS)
    assert not [a for a in cmd if a.startswith("--tools")]
    assert "--disallowedTools" in cmd and "--strict-mcp-config" in cmd
    assert len([r for r in caplog.records if "rejected --tools" in r.getMessage()]) == 1


@pytest.mark.parametrize(
    "case",
    [
        "allowedTools",
        "auth",
        "json-result-names-tools",
        "ran-then-stderr-names-tools",
        "stream-after-frames",
    ],
)
def test_unrelated_failure_does_not_flip(tmp_path, case):
    """C14: only a pre-execution rejection flips. Text a model, tool or web page
    could have written (a JSON `result`, output after frames) never does."""
    from brain.bridge import provider

    if case == "allowedTools":
        _failing_generate("", "error: unknown option '--allowedTools'")
    elif case == "auth":
        _failing_generate(
            '{"is_error": true, "result": "Failed to authenticate: OAuth session expired"}', ""
        )
    elif case == "json-result-names-tools":
        _failing_generate('{"is_error": true, "result": "please disable --tools now"}', "")
    elif case == "ran-then-stderr-names-tools":
        # The CLI ran (stdout carries its JSON), so this stderr isn't a parse
        # rejection even though it names --tools.
        _failing_generate('{"is_error": true, "result": "x"}', _REJECTION)
    else:
        # A frame arrived first, so the CLI got past argument parsing: this
        # failure isn't a rejection, whatever its stderr says.
        frame = '{"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}'
        _failing_stream([frame + "\n"], _REJECTION, tmp_path)
    assert provider._TOOLS_FLAG_SUPPORTED is True


@pytest.mark.parametrize(
    "confirm",
    ["auth-error", "empty", "timeout"],
)
def test_unexpected_confirmation_is_indeterminate(monkeypatch, confirm):
    """C6(f): help lacks the option, and the parse-only confirmation answers
    something other than "Input must be provided" / "unknown option --tools".
    That's "can't tell": fall back for now, don't cache "unsupported"."""
    import subprocess

    from brain.bridge import provider

    def _run(argv, **kwargs):
        if "--help" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=_SENTINEL, stderr="")
        if confirm == "timeout":
            raise subprocess.TimeoutExpired(argv, 10)
        err = "Failed to authenticate: OAuth session expired" if confirm == "auth-error" else ""
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)

    monkeypatch.setattr(provider.subprocess, "run", _run)
    provider._reset_tools_flag_cache()
    assert provider._cli_supports_tools_flag() is False
    assert provider._TOOLS_FLAG_SUPPORTED is None
