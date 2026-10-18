"""Scope addition 3 (owner, 2026-10-18): `nell model set <model> [--persona NAME]` and the stream-json save-time probe.

The command applies the chat model with the picker's rules and the picker's check: an alias or a full ``claude-``
id (64 cap), the same stream-json probe within the same budget, the same specific error on a rejection and a plain
"saved unchecked" outcome. With the persona's bridge running it goes through the picker's own settings endpoint
(hot swap, shared single-flight lock); otherwise it runs the check itself and writes the config.

Real files, the real ``brain.cli`` entry and the real endpoint (reached through a TestClient adapter in place of
urllib); only the Claude subprocess is faked, with the scrubbed real frames from tests/bridge/fixtures.
"""

from __future__ import annotations

import io
import json
import subprocess
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace
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


class _Resp:
    def __init__(self, content: bytes):
        self._content = content

    def read(self) -> bytes:
        return self._content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Bridge:
    """Stands in for urllib against a real in-process app: records the request, answers through the TestClient."""

    def __init__(self, client: TestClient, forced: tuple[int, bytes] | None = None):
        self.client = client
        self.requests: list[dict] = []
        self.forced = forced  # (status, body) answered instead of the real endpoint

    def urlopen(self, req, timeout=None):
        path = req.full_url.split("127.0.0.1:", 1)[1].split("/", 1)[1]
        self.requests.append(
            {"url": req.full_url, "path": "/" + path, "method": req.get_method(), "auth": req.get_header("Authorization"), "body": json.loads(req.data), "timeout": timeout}
        )
        if self.forced is not None:
            status, content = self.forced
            raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(content))
        resp = self.client.post("/" + path, content=req.data, headers=dict(req.header_items()))
        if resp.status_code >= 400:
            raise urllib.error.HTTPError(req.full_url, resp.status_code, "err", {}, io.BytesIO(resp.content))
        return _Resp(resp.content)


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
            with patch("urllib.request.urlopen", bridge.urlopen), direct:
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
            with patch("urllib.request.urlopen", bridge.urlopen):
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
    with patch("urllib.request.urlopen", side_effect=AssertionError("no bridge call for an invalid value")):
        rc = cli.main(["model", "set", "Fable", "--persona", "bob"])
    assert rc == 1 and MODEL_ID_RULE in capsys.readouterr().err and p.read_bytes() == before


def _via_bridge(monkeypatch, home: Path, run: _Run, capsys, model=CHAT, forced=None):
    cli._build_parser()
    with patch("subprocess.run", run), patch("subprocess.Popen", side_effect=_no_popen):
        p, client = _live_bridge(monkeypatch, home, run)
        with client as c:
            with patch("urllib.request.urlopen", _Bridge(c, forced).urlopen):
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
    with patch("urllib.request.urlopen", side_effect=AssertionError("refuse, do not call")), patch(
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
    with patch("urllib.request.urlopen", side_effect=AssertionError("no bridge call")), patch(
        "subprocess.run", _Run(lambda m: _ok())
    ), patch("subprocess.Popen", side_effect=_no_popen):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    assert rc == 0 and json.loads(p.read_text())["model"] == CHAT


def test_a_live_pid_whose_port_refuses_exits_1_and_writes_nothing(monkeypatch, home: Path, capsys):
    p = _write(home)
    before = p.read_bytes()
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: SimpleNamespace(pid=4321, port=51234, auth_token="t"))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    cli._build_parser()
    run = _Run(lambda m: _ok())
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), patch("subprocess.run", run):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and "could not reach" in err and p.read_bytes() == before and run.models == []
    assert "may still be starting" in err and "try again in a few seconds" in err and "nell service stop" in err
    assert "—" not in err


@pytest.mark.parametrize("how", ["read", "connect"])
def test_a_client_timeout_says_the_outcome_is_unknown(monkeypatch, home: Path, capsys, how):
    p = _write(home)
    before = p.read_bytes()
    monkeypatch.setattr("brain.bridge.state_file.read", lambda d: SimpleNamespace(pid=4321, port=51234, auth_token="t"))
    monkeypatch.setattr("brain.bridge.state_file.pid_is_alive", lambda pid: True)
    cli._build_parser()
    boom = TimeoutError("timed out") if how == "read" else urllib.error.URLError(TimeoutError("timed out"))
    with patch("urllib.request.urlopen", side_effect=boom):
        rc = cli.main(["model", "set", CHAT, "--persona", "bob"])
    err = capsys.readouterr().err
    assert rc == 1 and p.read_bytes() == before
    assert "outcome is unknown" in err and "may still apply" in err and "nell model show" in err
    assert "may still be starting" not in err and "—" not in err


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
