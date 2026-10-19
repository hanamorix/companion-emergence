"""Scope addition 3 (owner, 2026-10-18): `nell model set <model> [--persona NAME]` and the stream-json save-time probe.

The command applies the chat model with the picker's rules and the picker's check: an alias or a full ``claude-``
id (64 cap), the same stream-json probe within the same budget, the same specific error on a rejection and a plain
"saved unchecked" outcome. With the persona's bridge running it goes through the picker's own settings endpoint
(hot swap, shared single-flight lock); otherwise it runs the check itself and writes the config.

Real files, the real ``brain.cli`` entry and the real endpoint (reached through a TestClient adapter in place of
http.client); only the Claude subprocess is faked, with the scrubbed real frames from tests/bridge/fixtures.
"""

from __future__ import annotations

import http.client
import io
import json
import shlex
import socket
import struct
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain import cli
from brain.bridge import model_error
from brain.bridge import server as server_mod
from brain.bridge.provider import ClaudeCliProvider
from brain.bridge.server import build_app
from brain.persona_config import MODEL_ID_RULE, PersonaConfig

FIXDIR = Path(__file__).parent / "fixtures"
F302 = [json.loads(x) for x in (FIXDIR / "cli_2_1_302_model_not_found.ndjson").read_text(encoding="utf-8").splitlines() if x.strip()]
CHAT = "claude-sonnet-5"
DERIVED = "claude-haiku-5"
REWORDED = "Model unknown."


@pytest.fixture(autouse=True)
def _reset():
    model_error.reset()
    yield
    model_error.reset()


def _reject_ndjson() -> str:
    """The 2.1.302 rejection frames with the sentence reworded away: only the status and the code identify it."""
    out = json.loads(json.dumps(F302))
    for f in out:
        if f["type"] == "result":
            f["result"] = REWORDED
        else:
            f["message"]["content"] = [{"type": "text", "text": REWORDED}]
    return "".join(json.dumps(f) + "\n" for f in out)


def _proc(rc: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def _ok():
    return _proc(0, json.dumps({"type": "result", "is_error": False, "result": "OK"}))


def _reject():
    return _proc(1, _reject_ndjson())


def _quota():
    return _proc(1, json.dumps({"type": "result", "is_error": True, "api_error_status": 429, "result": "You've hit your session limit"}))


class _Run:
    def __init__(self, respond):
        self.respond = respond
        self.models: list[str] = []
        self.cmds: list[list[str]] = []
        self._lock = threading.Lock()

    def __call__(self, cmd, **kw):
        model = cmd[cmd.index("--model") + 1] if "--model" in cmd else ""
        with self._lock:
            self.models.append(model)
            self.cmds.append(list(cmd))
        return self.respond(model)


def _no_popen(*a, **k):
    raise AssertionError("the save-time check must not use the streaming Popen path")


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    d = tmp_path / "personas" / "bob"
    d.mkdir(parents=True)
    return d


def _write(d: Path, model="sonnet", provider="claude-cli", **extra) -> Path:
    p = d / "persona_config.json"
    p.write_text(json.dumps({"provider": provider, "searcher": "noop", "model": model, "user_name": "Bob", **extra}))
    return p


def _set(argv: list[str], run: _Run | None = None, capsys=None):
    cli._build_parser()  # import the lazily-imported modules before the CLI guards go on
    run = run or _Run(lambda m: _ok())
    with patch("subprocess.run", run), patch("subprocess.Popen", side_effect=_no_popen):
        rc = cli.main(["model", "set", *argv, "--persona", "bob"])
    out = capsys.readouterr() if capsys else None
    return rc, run, out


# --- MS4: validation ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["Sonnet", "fable", "claude-", "claude-" + "a" * 60, "claude sonnet-5", "claude-sonnet-5\n", "gpt-4", "sonnet[1m]"]
)
def test_unsupported_value_is_refused_with_the_clear_message(home: Path, capsys, bad):
    p = _write(home)
    before = p.read_bytes()
    rc, run, out = _set([bad], capsys=capsys)
    assert rc == 1
    assert MODEL_ID_RULE in out.err and repr(bad) in out.err
    assert p.read_bytes() == before and run.models == []


def test_unknown_persona_exits_1_without_writing(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    (tmp_path / "personas").mkdir()
    rc, run, out = _set(["opus"], capsys=capsys)
    assert rc == 1 and "not found" in out.err and run.models == []
    assert not (tmp_path / "personas" / "bob").exists()


# --- MS5: the bridge is not running ---------------------------------------------------------------------------


def test_alias_is_saved_clears_the_marker_and_the_list_without_any_check(home: Path, capsys):
    p = _write(home, model="fable", derived_unavailable=[DERIVED])
    assert PersonaConfig.load(p).model_unsupported == "fable"
    rc, run, out = _set(["opus"], capsys=capsys)
    assert rc == 0 and run.models == []
    saved = json.loads(p.read_text())
    assert saved["model"] == "opus" and "derived_unavailable" not in saved and saved["user_name"] == "Bob"
    assert PersonaConfig.load(p).model_unsupported is None
    assert "opus" in out.out


def test_the_fallback_alias_itself_clears_the_marker(home: Path, capsys):
    p = _write(home, model="fable")
    rc, _, _ = _set(["sonnet"], capsys=capsys)
    assert rc == 0 and json.loads(p.read_text())["model"] == "sonnet"
    assert PersonaConfig.load(p).model_unsupported is None


def test_full_id_is_checked_with_the_stream_json_probe_then_saved(home: Path, capsys):
    p = _write(home)
    rc, run, out = _set([CHAT], capsys=capsys)
    assert rc == 0
    assert sorted(set(run.models)) == [DERIVED, CHAT]  # the chat id, then the derived tier id
    for cmd in run.cmds:
        i = cmd.index("--output-format")
        assert cmd[i + 1] == "stream-json" and "--verbose" in cmd
        assert "--mcp-config" not in cmd and "--allowedTools" not in cmd
    assert json.loads(p.read_text())["model"] == CHAT
    assert "checked" in out.out and "—" not in out.out


def test_a_rejected_chat_id_is_refused_from_the_code_alone(home: Path, capsys):
    p = _write(home)
    before = p.read_bytes()
    rc, run, out = _set([CHAT], _Run(lambda m: _reject()), capsys=capsys)
    assert rc == 1 and p.read_bytes() == before
    assert CHAT in out.err and "may not exist" in out.err and "Nothing was changed" in out.err
    assert run.models == [CHAT]  # no derived probe is spent on a refused id
    assert "—" not in out.err


def test_a_rejected_derived_id_is_saved_with_a_fallback_line(home: Path, capsys):
    p = _write(home)
    rc, _, out = _set([CHAT], _Run(lambda m: _reject() if m == DERIVED else _ok()), capsys=capsys)
    assert rc == 0
    saved = json.loads(p.read_text())
    assert saved["model"] == CHAT and saved["derived_unavailable"] == [DERIVED]
    assert f'"{DERIVED}"' in out.out and "could not be used" in out.out and "—" not in out.out


def test_an_inconclusive_check_saves_unchecked_and_says_so(home: Path, capsys):
    p = _write(home)
    rc, _, out = _set([CHAT], _Run(lambda m: _quota()), capsys=capsys)
    assert rc == 0 and json.loads(p.read_text())["model"] == CHAT
    assert "saved unchecked" in out.out and "—" not in out.out


def test_a_non_claude_cli_persona_is_saved_without_any_check(home: Path, capsys):
    p = _write(home, provider="fake")
    rc, run, _ = _set([CHAT], capsys=capsys)
    assert rc == 0 and run.models == [] and json.loads(p.read_text())["model"] == CHAT


def test_a_full_id_clears_the_unsupported_marker(home: Path, capsys):
    p = _write(home, model="fable")
    rc, _, _ = _set([CHAT], capsys=capsys)
    assert rc == 0 and json.loads(p.read_text())["model"] == CHAT
    assert PersonaConfig.load(p).model_unsupported is None


# --- MS6: the bridge is running ----------------------------------------------------------------------------------


class _FakeResponse:
    """What HTTPConnection.getresponse() returns: a status and a readable body (bytes, or any object with read(n))."""

    def __init__(self, status: int, content: Any):
        self.status = status
        self._fp = io.BytesIO(content) if isinstance(content, (bytes, bytearray)) else content

    def read(self, n: int = -1) -> bytes:
        return self._fp.read(n)


class _FakeConn:
    def __init__(self, bridge: _Bridge, host: str, port: int, timeout):
        self.bridge, self.host, self.port, self.timeout = bridge, host, port, timeout

    def connect(self) -> None:
        self.bridge.connects += 1
        if self.bridge.connect_exc is not None:
            raise self.bridge.connect_exc

    def request(self, method, url, body=None, headers=None) -> None:
        headers = headers or {}
        self.bridge.requests.append(
            {
                "url": f"http://{self.host}:{self.port}{url}",
                "path": url,
                "method": method,
                "auth": headers.get("Authorization"),
                "body": json.loads(body),
                "timeout": self.timeout,
            }
        )
        self._body, self._headers, self._path = body, headers, url
        if self.bridge.request_exc is not None:
            raise self.bridge.request_exc

    def getresponse(self) -> _FakeResponse:
        if self.bridge.response_exc is not None:
            raise self.bridge.response_exc
        if self.bridge.forced is not None:
            return _FakeResponse(*self.bridge.forced)
        resp = self.bridge.client.post(self._path, content=self._body, headers=self._headers)
        return _FakeResponse(resp.status_code, resp.content)

    def close(self) -> None:
        pass


class _Bridge:
    """Stands in for http.client.HTTPConnection against a real in-process app: records the request, answers
    through the TestClient (or a forced (status, body)), and can fail at any phase: connect, request, response."""

    def __init__(
        self,
        client: TestClient | None = None,
        forced: tuple[int, Any] | None = None,
        *,
        connect_exc: BaseException | None = None,
        request_exc: BaseException | None = None,
        response_exc: BaseException | None = None,
    ):
        self.client = client
        self.forced = forced
        self.connect_exc, self.request_exc, self.response_exc = connect_exc, request_exc, response_exc
        self.requests: list[dict] = []
        self.connects = 0

    def factory(self, host, port, timeout=None) -> _FakeConn:
        return _FakeConn(self, host, port, timeout)


def _no_bridge_call():
    """patch() for http.client.HTTPConnection that fails the test if the command makes any bridge connection."""
    return patch("http.client.HTTPConnection", side_effect=AssertionError("no bridge connection may be made"))


def _live_bridge(monkeypatch, home: Path, run: _Run):
    p = _write(home)
    monkeypatch.setattr("brain.bridge.provider.get_provider", lambda *a, **k: ClaudeCliProvider(model="sonnet"))
    client = TestClient(build_app(persona_dir=home, client_origin="tests"))
    state = SimpleNamespace(pid=4321, port=51234, auth_token="tok-123")
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: state)
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: pid == 4321)
    return p, client


def test_a_running_bridge_gets_the_request_and_switches_live(monkeypatch, home: Path, capsys):
    run = _Run(lambda m: _ok())
    cli._build_parser()
    with patch("subprocess.run", run), patch("subprocess.Popen", side_effect=_no_popen):
        p, client = _live_bridge(monkeypatch, home, run)
        with client as c:
            bridge = _Bridge(c)
            direct = patch.object(cli, "_model_set_direct", side_effect=AssertionError("must go through the bridge"))
            with patch("http.client.HTTPConnection", bridge.factory), direct:
                rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
            hot = c.app.state.bridge.provider._model
    out = capsys.readouterr().out
    assert rc == 0 and hot == CHAT  # the picker's endpoint hot-swapped the running provider
    assert len(bridge.requests) == 1
    r = bridge.requests[0]
    assert r["path"] == "/persona/config/model" and r["method"] == "POST" and r["body"] == {"model": CHAT}
    assert r["auth"] == "Bearer tok-123" and r["url"].startswith("http://127.0.0.1:51234/")
    assert r["timeout"] and r["timeout"] > server_mod._check_budget_s()
    assert json.loads(p.read_text())["model"] == CHAT and "checked" in out


def test_a_running_bridge_rejection_is_reported_from_its_400(monkeypatch, home: Path, capsys):
    run = _Run(lambda m: _reject())
    cli._build_parser()
    with patch("subprocess.run", run), patch("subprocess.Popen", side_effect=_no_popen):
        p, client = _live_bridge(monkeypatch, home, run)
        before = p.read_bytes()
        with client as c:
            bridge = _Bridge(c)
            with patch("http.client.HTTPConnection", bridge.factory):
                rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
            after = p.read_bytes()
    err = capsys.readouterr().err
    assert rc == 1 and CHAT in err and "Nothing was changed" in err
    assert json.loads(after)["model"] == "sonnet" and json.loads(before)["model"] == "sonnet"


def test_an_unsupported_value_never_reaches_a_running_bridge(monkeypatch, home: Path, capsys):
    p = _write(home)
    before = p.read_bytes()
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: SimpleNamespace(pid=4321, port=51234, auth_token="t"))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    cli._build_parser()
    with _no_bridge_call():
        rc = cli.main(["model", "set", "Fable", "--persona", "bob"])
    assert rc == 1 and MODEL_ID_RULE in capsys.readouterr().err and p.read_bytes() == before


def _via_bridge(monkeypatch, home: Path, run: _Run, capsys, model=CHAT, forced=None):
    cli._build_parser()
    with patch("subprocess.run", run), patch("subprocess.Popen", side_effect=_no_popen):
        p, client = _live_bridge(monkeypatch, home, run)
        with client as c:
            with patch("http.client.HTTPConnection", _Bridge(c, forced).factory):
                rc = cli.main(["model", "set", model, "--persona", "bob"])
    return rc, p, capsys.readouterr()


def test_a_running_bridge_that_could_not_check_prints_the_unchecked_lines(monkeypatch, home: Path, capsys):
    rc, p, out = _via_bridge(monkeypatch, home, _Run(lambda m: _quota()), capsys)
    assert rc == 0 and json.loads(p.read_text())["model"] == CHAT
    assert "saved unchecked" in out.out and "not checked yet" in out.out and DERIVED in out.out
    assert "the running bridge switched to it now" in out.out and "—" not in out.out


def test_a_running_bridge_that_found_a_fallback_prints_the_fallback_line(monkeypatch, home: Path, capsys):
    rc, p, out = _via_bridge(monkeypatch, home, _Run(lambda m: _reject() if m == DERIVED else _ok()), capsys)
    assert rc == 0 and json.loads(p.read_text())["derived_unavailable"] == [DERIVED]
    assert f'"{DERIVED}"' in out.out and "could not be used" in out.out and "family" in out.out


@pytest.mark.parametrize(
    "status,body,expect",
    [
        (401, {"detail": "Unauthorized"}, "Unauthorized"),
        (500, {"detail": "boom"}, "boom"),
        (422, {"detail": [{"msg": "bad"}]}, "bad"),
        (500, ["not", "a", "dict"], "not"),
        (502, "plain text", "plain text"),
    ],
)
def test_a_bridge_error_status_is_reported_as_the_bridges_answer_not_as_an_invalid_model(monkeypatch, home: Path, capsys, status, body, expect):
    rc, p, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(status, json.dumps(body).encode()))
    assert rc == 1
    assert f"the bridge answered {status}" in out.err and expect in out.err
    assert "not a valid model" not in out.err and MODEL_ID_RULE not in out.err
    assert json.loads(p.read_text())["model"] == "sonnet"  # a bridge error changes nothing


def test_an_unreadable_error_body_still_names_the_status(monkeypatch, home: Path, capsys):
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(503, b"<html>down</html>"))
    assert rc == 1 and "the bridge answered 503" in out.err and "not a valid model" not in out.err


def test_a_bridge_that_started_while_the_check_ran_gets_nothing_written_behind_it(monkeypatch, home: Path, capsys):
    p = _write(home)
    before = p.read_bytes()
    states = iter([None, SimpleNamespace(pid=4321, port=51234, auth_token="t")])
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: next(states))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    run = _Run(lambda m: _ok())
    cli._build_parser()
    with _no_bridge_call(), patch(
        "subprocess.run", run
    ), patch("subprocess.Popen", side_effect=_no_popen):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and p.read_bytes() == before
    assert "started while" in err and "run the command again" in err.lower() and "—" not in err
    assert run.models  # the check really ran before the second look


def test_a_bridge_that_started_before_an_alias_write_is_also_refused(monkeypatch, home: Path, capsys):
    p = _write(home)
    before = p.read_bytes()
    states = iter([None, SimpleNamespace(pid=4321, port=51234, auth_token="t")])
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: next(states))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    cli._build_parser()
    rc = cli.main(["model", "set", "opus", "--persona", "bob"])
    assert rc == 1 and p.read_bytes() == before and "started while" in capsys.readouterr().err


def test_a_stale_state_file_means_not_running(monkeypatch, home: Path, capsys):
    p = _write(home)
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: SimpleNamespace(pid=999999, port=51234, auth_token="t"))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: False)
    cli._build_parser()
    with _no_bridge_call(), patch(
        "subprocess.run", _Run(lambda m: _ok())
    ), patch("subprocess.Popen", side_effect=_no_popen):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    assert rc == 0 and json.loads(p.read_text())["model"] == CHAT


def _closed_port() -> int:
    """A loopback port nothing listens on (the OS refuses connections to it), without any error number."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_a_live_pid_whose_port_refuses_exits_1_and_writes_nothing(monkeypatch, home: Path, capsys):
    p = _write(home)
    before = p.read_bytes()
    _point_at(monkeypatch, _closed_port())
    run = _Run(lambda m: _ok())
    with patch("subprocess.run", run):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and "could not reach" in err and p.read_bytes() == before and run.models == []
    assert "may still be starting" in err and "try again in a few seconds" in err
    assert "nell supervisor stop --persona bob" in err and "nell service stop" not in err
    assert "—" not in err
    # the command the message suggests must be a real one: it parses with the real parser
    suggested = "nell supervisor stop --persona bob"
    parsed = cli._build_parser().parse_args(shlex.split(suggested)[1:])
    from brain.bridge.daemon import cmd_stop

    assert parsed.command == "supervisor" and parsed.persona == "bob" and parsed.func is cmd_stop


@pytest.mark.parametrize("phase", ["request", "response", "read"])
def test_a_client_timeout_after_connecting_says_the_outcome_is_unknown(monkeypatch, home: Path, capsys, phase):
    p = _write(home)
    before = p.read_bytes()
    _point_at(monkeypatch, 51234)
    boom = TimeoutError("timed out")
    if phase == "read":
        bridge = _Bridge(forced=(200, _GreedyResp(b"", fail=boom)))
    else:
        bridge = _Bridge(**{"request_exc" if phase == "request" else "response_exc": boom})
    with patch("http.client.HTTPConnection", bridge.factory):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and p.read_bytes() == before
    assert "outcome is unknown" in err and "may still apply" in err and "nell model show" in err
    assert "may still be starting" not in err and "could not reach" not in err and "—" not in err


# --- MS7: one implementation ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["ok", "unchecked", "fallback", "rejected", "alias"],
)
def test_endpoint_and_direct_mode_return_the_same_body(tmp_path: Path, monkeypatch, case):
    import asyncio

    respond = {
        "ok": lambda m: _ok(),
        "unchecked": lambda m: _quota(),
        "fallback": lambda m: _reject() if m == DERIVED else _ok(),
        "rejected": lambda m: _reject(),
        "alias": lambda m: _ok(),
    }[case]
    model = "opus" if case == "alias" else CHAT
    bodies = []
    for mode in ("direct", "endpoint"):
        d = tmp_path / mode
        d.mkdir()
        _write(d)
        with patch("subprocess.run", _Run(respond)), patch("subprocess.Popen", side_effect=_no_popen):
            if mode == "direct":
                status, body = asyncio.run(
                    server_mod._apply_chat_model(d, model, check=True, lock=asyncio.Lock(), hot_swap=None)
                )
            else:
                monkeypatch.setattr("brain.bridge.provider.get_provider", lambda *a, **k: ClaudeCliProvider(model="sonnet"))
                with TestClient(build_app(persona_dir=d, client_origin="tests")) as c:
                    r = c.post("/persona/config/model", json={"model": model})
                    status, body = r.status_code, r.json()
        bodies.append((status, body))
        model_error.reset()
    assert bodies[0] == bodies[1], bodies
    assert bodies[0][0] == (400 if case == "rejected" else 200)


# --- hostile or odd bridge answers (round-3 review) ----------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [b"not json at all", b"[1, 2]", b'"ok"', b"{}", b'{"ok": false, "model": "claude-sonnet-5"}', b'{"model": "claude-sonnet-5"}', b""],
    ids=["not-json", "list", "string", "empty-object", "ok-false", "no-ok-key", "empty-body"],
)
def test_a_200_that_is_not_the_endpoints_success_body_is_an_error_not_a_success(monkeypatch, home: Path, capsys, content):
    rc, p, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, content))
    assert rc == 1
    assert "the bridge answered 200" in out.err and "unexpected" in out.err
    assert "chat model is now" not in out.out and "Traceback" not in out.err
    assert json.loads(p.read_text())["model"] == "sonnet"


def test_a_200_with_ok_true_still_succeeds(monkeypatch, home: Path, capsys):
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, b'{"ok": true, "model": "claude-sonnet-5"}'))
    assert rc == 0 and "chat model is now claude-sonnet-5." in out.out


class _GreedyBody(io.BytesIO):
    """An error body that records how much the caller asked to read, and can fail to read."""

    def __init__(self, data: bytes, fail: bool = False):
        super().__init__(data)
        self.asked: list[int | None] = []
        self.fail = fail

    def read(self, n=-1):
        self.asked.append(n if n != -1 else None)
        if self.fail:
            raise OSError("connection reset")
        return super().read(n)


def test_an_error_body_is_read_with_a_cap(monkeypatch, home: Path, capsys):
    body = _GreedyBody(b'{"detail": "' + b"x" * 5_000_000 + b'"}')
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(500, body))
    assert rc == 1 and "the bridge answered 500" in out.err
    assert body.asked and all(n is not None and n <= 65536 for n in body.asked), body.asked
    assert len(out.err) < 1000


def test_an_error_body_that_cannot_be_read_still_names_the_status(monkeypatch, home: Path, capsys):
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(500, _GreedyBody(b"x", fail=True)))
    assert rc == 1 and "the bridge answered 500" in out.err and "Traceback" not in out.err


def test_control_characters_from_the_bridge_are_never_printed(monkeypatch, home: Path, capsys):
    hostile = '\x1b[31mred\x07 \x1b]0;title\x07line\nbreak'
    body = json.dumps({"detail": hostile}).encode()
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(401, body))
    assert rc == 1 and "the bridge answered 401" in out.err
    for bad in ("\x1b", "\x07"):
        assert bad not in out.err
    assert out.err.count("\n") == 1  # only the final newline of the line itself


def test_control_characters_in_a_success_body_are_never_printed(monkeypatch, home: Path, capsys):
    ok = {
        "ok": True,
        "model": CHAT,
        "checked": True,
        "fallbacks": [{"family": "ha\x1b[2Jiku", "wanted": "claude-haiku-5\x07", "using": "ha\niku"}],
        "tiers_unchecked": ["claude-\x1b]0;x\x07haiku-5"],
    }
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, json.dumps(ok).encode()))
    assert rc == 0 and "\x1b" not in out.out and "\x07" not in out.out
    assert "claude-haiku-5" in out.out


# --- round-4 recommendations ------------------------------------------------------------------------------------


class _GreedyResp:
    """A 200 response that records how much the caller asked to read."""

    def __init__(self, data: bytes, fail: Exception | None = None):
        self._buf = io.BytesIO(data)
        self.asked: list[int | None] = []
        self.fail = fail

    def read(self, n=-1):
        self.asked.append(n if n != -1 else None)
        if self.fail:
            raise self.fail
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_200_body_is_read_with_a_cap(monkeypatch, home: Path, capsys):
    resp = _GreedyResp(b'{"ok": true, "model": "' + b"x" * 5_000_000 + b'"}')
    rc, p, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, resp))
    assert resp.asked and all(n is not None and n <= 65536 for n in resp.asked), resp.asked
    assert rc == 1 and "unexpected body" in out.err and json.loads(p.read_text())["model"] == "sonnet"


@pytest.mark.parametrize(
    "extra,absent",
    [
        ({"fallbacks": "boom"}, ["boom"]),
        ({"fallbacks": {"family": "haiku", "wanted": "w", "using": "u"}}, ["family"]),
        ({"tiers_unchecked": "abc"}, ["a, b, c", "not checked yet"]),
        ({"tiers_unchecked": {"x": 1}}, ["not checked yet"]),
        ({"tiers_unchecked": []}, ["not checked yet"]),
    ],
    ids=["fallbacks-string", "fallbacks-dict", "unchecked-string", "unchecked-dict", "unchecked-empty"],
)
def test_odd_shapes_for_fallbacks_and_unchecked_ids_are_ignored(monkeypatch, home: Path, capsys, extra, absent):
    body = json.dumps({"ok": True, "model": CHAT, **extra}).encode()
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, body))
    assert rc == 0 and "chat model is now" in out.out and "Traceback" not in out.err
    for text in absent:
        assert text not in out.out, (text, out.out)


def test_non_dict_fallback_entries_are_skipped_and_a_good_one_still_prints(monkeypatch, home: Path, capsys):
    good = {"family": "haiku", "wanted": DERIVED, "using": "haiku"}
    body = json.dumps({"ok": True, "model": CHAT, "fallbacks": [1, "x", None, ["y"], good]}).encode()
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, body))
    assert rc == 0 and out.out.count("could not be used") == 1 and f'"{DERIVED}"' in out.out


@pytest.mark.parametrize("detail", ["\x1b[31mred\x07", "line\x1b]0;t\x07"])
def test_the_detail_of_an_unexpected_200_is_cleaned(monkeypatch, home: Path, capsys, detail):
    for content in (json.dumps({"ok": False, "detail": detail}).encode(), json.dumps(detail).encode()):
        rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, content))
        assert rc == 1 and "unexpected body" in out.err
        assert "\x1b" not in out.err and "\x07" not in out.err and out.err.count("\n") == 1


def test_the_reason_of_an_unreachable_port_is_cleaned(monkeypatch, home: Path, capsys):
    _write(home)
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: SimpleNamespace(pid=4321, port=51234, auth_token="t"))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    cli._build_parser()
    bridge = _Bridge(connect_exc=OSError("refused\x1b[31m\nred\x07"))
    with patch("http.client.HTTPConnection", bridge.factory):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and "could not reach" in err and "\x1b" not in err and "\x07" not in err and err.count("\n") == 1


def test_newlines_and_tabs_from_the_bridge_become_spaces(monkeypatch, home: Path, capsys):
    body = json.dumps({"detail": "first line\nsecond\tthird"}).encode()
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(500, body))
    assert rc == 1 and "first line second third" in out.err and out.err.count("\n") == 1


@pytest.mark.parametrize("exc", [http.client.BadStatusLine("garbage"), http.client.IncompleteRead(b"x"), http.client.ImproperConnectionState("x")])
def test_a_non_http_answer_is_a_clear_message_not_a_traceback(monkeypatch, home: Path, capsys, exc):
    p = _write(home)
    before = p.read_bytes()
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: SimpleNamespace(pid=4321, port=51234, auth_token="t"))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    cli._build_parser()
    bridge = _Bridge(response_exc=exc)
    with patch("http.client.HTTPConnection", bridge.factory):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and p.read_bytes() == before and "Traceback" not in err
    assert "did not understand the bridge's answer" in err and "outcome is unknown" in err and "nell model show" in err


def test_a_truncated_200_read_is_a_clear_message(monkeypatch, home: Path, capsys):
    resp = _GreedyResp(b"", fail=http.client.IncompleteRead(b"par"))
    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(200, resp))
    assert rc == 1 and "did not understand the bridge's answer" in out.err and "Traceback" not in out.err


def test_an_error_body_whose_read_raises_an_http_exception_still_names_the_status(monkeypatch, home: Path, capsys):
    class _Truncated(io.BytesIO):
        def read(self, n=-1):
            raise http.client.IncompleteRead(b"x")

    rc, _, out = _via_bridge(monkeypatch, home, _Run(lambda m: _ok()), capsys, forced=(500, _Truncated(b"")))
    assert rc == 1 and "the bridge answered 500" in out.err and "Traceback" not in out.err


def _read_request(conn: socket.socket) -> bool:
    """Read one whole HTTP request (headers and the Content-Length body), so closing later cannot reset the client.
    False if the client hung up without sending anything (the command's pre-flight connect)."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return bool(data)
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    while len(rest) < length:
        chunk = conn.recv(65536)
        if not chunk:
            return True
        rest += chunk
    return True


def _raw_server(payload: bytes, *, reset: bool = False, accepted: list | None = None):
    """A TCP server on a free loopback port. A connection that hangs up without sending a request is ignored. For the connection that carries the request it reads the whole request first,
    so the client has finished sending, then either (reset=False) sends ``payload``, shuts down its write side
    CLEANLY and waits for the client to finish reading and close, so the client sees exactly ``payload`` and then
    end-of-stream; or (reset=True) aborts the connection with a TCP reset (SO_LINGER 0) without sending anything."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(10)

    def serve():
        try:
            while True:
                conn, _ = srv.accept()
                if accepted is not None:
                    accepted.append(1)
                conn.settimeout(10)
                try:
                    if not _read_request(conn):
                        continue  # nothing was sent on this connection: wait for a real request
                    if reset:
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                        return
                    conn.sendall(payload)
                    conn.shutdown(socket.SHUT_WR)
                    try:
                        while conn.recv(65536):  # wait until the client has read what it wants and closed
                            pass
                    except OSError:
                        pass
                    return
                finally:
                    conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


def _point_at(monkeypatch, port: int) -> None:
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: SimpleNamespace(pid=4321, port=port, auth_token="t"))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    cli._build_parser()


@pytest.mark.parametrize(
    "payload",
    [b"NOT HTTP AT ALL\r\n\r\n", b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhel"],
    ids=["not-http", "truncated-chunked"],
)
def test_a_real_socket_that_answers_badly_gives_a_clear_message(monkeypatch, home: Path, capsys, payload):
    p = _write(home)
    before = p.read_bytes()
    _point_at(monkeypatch, _raw_server(payload))
    rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and p.read_bytes() == before and "Traceback" not in err
    assert "did not understand the bridge's answer" in err and "nell model show" in err
    assert "could not reach" not in err


def test_a_real_connection_reset_after_the_request_was_sent_is_an_unknown_outcome(monkeypatch, home: Path, capsys):
    p = _write(home)
    before = p.read_bytes()
    _point_at(monkeypatch, _raw_server(b"", reset=True))
    rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and p.read_bytes() == before and "Traceback" not in err
    assert "outcome is unknown" in err and "may still apply" in err and "nell model show" in err
    assert "could not reach" not in err and "may still be starting" not in err


def test_a_real_closed_port_is_could_not_reach(monkeypatch, home: Path, capsys):
    p = _write(home)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now: the connection is refused before any request is sent
    _point_at(monkeypatch, port)
    rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and "could not reach the bridge" in err and "may still be starting" in err
    assert "outcome is unknown" not in err and json.loads(p.read_text())["model"] == "sonnet"


@pytest.mark.parametrize("phase", ["request", "response", "read"])
@pytest.mark.parametrize(
    "exc",
    [
        ConnectionResetError("connection reset"),
        BrokenPipeError("broken pipe"),
        ConnectionAbortedError("connection aborted"),
        OSError("input/output error"),
    ],
    ids=["reset", "broken-pipe", "aborted", "oserror"],
)
def test_an_error_after_connect_returned_is_an_unknown_outcome(monkeypatch, home: Path, capsys, exc, phase):
    """connect() succeeded, so a failure in request(), getresponse() or the read (whatever its type or OS error
    number) happens once the bridge may have the request: the outcome is unknown. No error numbers are used."""
    p = _write(home)
    before = p.read_bytes()
    _point_at(monkeypatch, 51234)
    if phase == "read":
        bridge = _Bridge(forced=(200, _GreedyResp(b"", fail=exc)))
    else:
        bridge = _Bridge(**{"request_exc" if phase == "request" else "response_exc": exc})
    with patch("http.client.HTTPConnection", bridge.factory):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and p.read_bytes() == before and bridge.connects == 1
    assert "outcome is unknown" in err and "nell model show" in err and "could not reach" not in err and "—" not in err


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionRefusedError("connection refused"),
        socket.gaierror("name not resolved"),
        OSError("network unreachable"),
        TimeoutError("connect timed out"),
    ],
    ids=["refused", "dns", "unreachable", "connect-timeout"],
)
def test_a_failure_in_connect_is_could_not_reach_and_nothing_is_sent(monkeypatch, home: Path, capsys, exc):
    """Decided by WHERE it failed (connect() itself, before any request is written), not by error number."""
    p = _write(home)
    _point_at(monkeypatch, 51234)
    bridge = _Bridge(connect_exc=exc)
    with patch("http.client.HTTPConnection", bridge.factory):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and "could not reach the bridge" in err and "outcome is unknown" not in err
    assert bridge.connects == 1 and bridge.requests == []  # nothing was written
    assert json.loads(p.read_text())["model"] == "sonnet"


def test_one_real_connection_carries_the_request_and_the_answer(monkeypatch, home: Path, capsys):
    """The bridge sees exactly ONE connection (no empty probe first) and the command succeeds on its answer."""
    p = _write(home)
    body = b'{"ok": true, "model": "claude-sonnet-5", "checked": true}'
    payload = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body
    accepted: list[int] = []
    _point_at(monkeypatch, _raw_server(payload, accepted=accepted))
    rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    out = capsys.readouterr()
    assert rc == 0 and "chat model is now claude-sonnet-5 (checked with Claude)." in out.out
    assert len(accepted) == 1, accepted
    assert json.loads(p.read_text())["model"] == "sonnet"  # the command asked the bridge; it wrote nothing itself
