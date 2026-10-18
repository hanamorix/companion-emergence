"""#440 - a Claude CLI model rejection surfaces as a specific error (chat), is checked at Apply time,
and is remembered for the ambient jobs. Real ClaudeCliProvider and real routes; only the subprocess is
faked, with the scrubbed frames the real CLI 2.1.219 emitted (tests/bridge/fixtures).

Fixture-replay: this proves the parser and the wire. That the CURRENT CLI still words the rejection this
way is the pre-flip check in test_model_rejection_live.py.
"""

from __future__ import annotations

import getpass
import io
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain.bridge import model_error, provider_auth
from brain.bridge.provider import ClaudeCliProvider
from brain.bridge.server import build_app
from brain.persona_config import PersonaConfig

FIXTURE = Path(__file__).parent / "fixtures" / "cli_2_1_219_model_not_found.ndjson"
FRAMES = [json.loads(x) for x in FIXTURE.read_text(encoding="utf-8").splitlines() if x.strip()]
# probe 1 (unknown id claude-sonet-5): frames[0] assistant, frames[1] result
REJECT_NDJSON = "".join(json.dumps(f) + "\n" for f in FRAMES[:2])
REJECT_RESULT = json.dumps(FRAMES[1])
BAD = "claude-sonet-5"
LIVE = "Failed to authenticate: OAuth session expired and could not be refreshed"


class _FakePopen:
    def __init__(self, lines):
        self.stdin = io.StringIO()
        self.stdout = iter(lines)
        self.stderr = io.StringIO("")
        self.pid = 4242

    def wait(self, timeout=None):
        return 1

    def poll(self):
        return 1

    def terminate(self):
        pass

    def kill(self):
        pass


def _proc(rc: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def _ok_json() -> str:
    return json.dumps({"is_error": False, "result": "OK"})


def _client(persona_dir: Path, monkeypatch, model: str = "sonnet") -> TestClient:
    monkeypatch.setattr("brain.bridge.provider.get_provider", lambda *a, **k: ClaudeCliProvider(model=model))
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


class _Run:
    """A recording stand-in for subprocess.run: ``respond(model) -> CompletedProcess``."""

    def __init__(self, respond, delay: float = 0.0, watch: frozenset = frozenset()):
        self.respond = respond
        self.watch = watch
        self.delay = delay
        self.models: list[str] = []
        self._lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    def __call__(self, cmd, **kw):
        model = cmd[cmd.index("--model") + 1] if "--model" in cmd else ""
        watched = model in self.watch
        with self._lock:
            self.models.append(model)
            if watched:
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self.respond(model)
        finally:
            if watched:
                with self._lock:
                    self.inflight -= 1

    def calls_for(self, model: str) -> int:
        return self.models.count(model)


def _respond_reject_bad(model: str):
    return _proc(1, REJECT_RESULT) if model == BAD else _proc(0, _ok_json())


# --- C7: chat error surfaces (fixture replay through the real provider) -----------------------------


def test_http_chat_returns_model_unavailable_with_the_model(persona_dir: Path, monkeypatch):
    with _client(persona_dir, monkeypatch) as c:
        sid = c.post("/session/new", json={}).json()["session_id"]
        with patch("subprocess.Popen", return_value=_FakePopen([json.dumps(f) + "\n" for f in FRAMES[:2]])), \
             patch("subprocess.run", return_value=_proc(1, REJECT_NDJSON)):
            r = c.post("/chat", json={"session_id": sid, "message": "hello"})
    assert r.status_code == 502
    assert r.json()["detail"] == f"model_unavailable:{BAD}"


def test_ws_error_frame_carries_code_and_model(persona_dir: Path, monkeypatch):
    with _client(persona_dir, monkeypatch) as c:
        sid = c.post("/session/new", json={}).json()["session_id"]
        with patch("subprocess.Popen", return_value=_FakePopen([json.dumps(f) + "\n" for f in FRAMES[:2]])), \
             patch("subprocess.run", return_value=_proc(1, REJECT_NDJSON)):
            with c.websocket_connect(f"/stream/{sid}") as ws:
                ws.send_json({"message": "hello"})
                frames = []
                while True:
                    f = ws.receive_json()
                    frames.append(f)
                    if f.get("type") in ("error", "done"):
                        break
    assert frames[-1] == {"type": "error", "code": "model_unavailable", "model": BAD, "done": True}


def test_chat_error_code_classification_and_precedence():
    from brain.bridge import server as server_mod

    rej = FRAMES[1]["result"]
    assert server_mod._chat_error_code(RuntimeError(f"[claude_cli_error] {rej}")) == "model_unavailable"
    assert server_mod._chat_error_code(RuntimeError(f"[claude_cli_exit] exit 1: api_error_status=404; is_error=True; {rej}")) == "model_unavailable"
    # auth wins, 429 stays generic, as before
    assert server_mod._chat_error_code(RuntimeError(f"{LIVE}; {rej}")) == "auth_expired"
    assert server_mod._chat_error_code(RuntimeError("[claude_cli_exit] exit 1: api_error_status=429")) == "provider_failed"
    assert server_mod._chat_error_model(RuntimeError(rej), ClaudeCliProvider(model="sonnet")) == BAD
    assert server_mod._chat_error_model(RuntimeError("[claude_cli_exit] exit 1: api_error_status=429"), ClaudeCliProvider(model="sonnet")) is None


# --- C14: check on save -------------------------------------------------------------------------


def _post_model(c: TestClient, model: str):
    return c.post("/persona/config/model", json={"model": model})


def _saved_model(persona_dir: Path) -> str:
    return PersonaConfig.load(persona_dir / "persona_config.json").model


def test_rejected_custom_id_is_refused_and_leaves_no_trace(persona_dir: Path, monkeypatch, caplog):
    run = _Run(_respond_reject_bad)
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        provider = c.app.state.bridge.provider
        with caplog.at_level(logging.WARNING):
            r = _post_model(c, BAD)
        state = c.get("/persona/state").json()
        assert r.status_code == 400, r.text
        assert r.json() == {"ok": False, "error": "model_unavailable", "model": BAD}
        assert _saved_model(persona_dir) == "sonnet"
        assert provider._model == "sonnet"
        # C14h: nothing recorded or logged for an id that was never saved
        assert state["model_rejections"] == []
        assert model_error.state() == []
        assert not [rec for rec in caplog.records if BAD in rec.getMessage() and rec.name.endswith("model_error")]
    assert run.calls_for(BAD) == 1  # C14g: exactly one check call


def test_accepted_custom_id_is_saved_hot_swapped_and_marked_checked(persona_dir: Path, monkeypatch):
    run = _Run(lambda m: _proc(0, _ok_json()))
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        provider = c.app.state.bridge.provider
        r = _post_model(c, "claude-sonnet-5")
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "model": "claude-sonnet-5", "checked": True}
        assert provider._model == "claude-sonnet-5"
    assert _saved_model(persona_dir) == "claude-sonnet-5"
    assert run.calls_for("claude-sonnet-5") >= 1


def test_check_call_writes_no_usage_row(persona_dir: Path, monkeypatch):
    run = _Run(lambda m: _proc(0, _ok_json()))
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        _post_model(c, "claude-sonnet-5")
    usage = persona_dir / "chat_usage.jsonl"
    rows = [json.loads(x) for x in usage.read_text().splitlines()] if usage.exists() else []
    assert not [r for r in rows if r.get("model") == "claude-sonnet-5"]


def test_non_rejection_failure_saves_with_checked_false(persona_dir: Path, monkeypatch, caplog):
    quota = json.dumps({"is_error": True, "api_error_status": 429, "result": "You've hit your session limit"})
    run = _Run(lambda m: _proc(1, quota))
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        with caplog.at_level(logging.INFO):
            r = _post_model(c, "claude-sonnet-5")
    assert r.status_code == 200, r.text
    # the derived tier id could not be probed either (same failure), so it is reported as not checked yet
    assert r.json() == {"ok": True, "model": "claude-sonnet-5", "checked": False, "tiers_unchecked": ["claude-haiku-5"]}
    assert _saved_model(persona_dir) == "claude-sonnet-5"
    assert any("reason=failed" in rec.getMessage() for rec in caplog.records)


def test_unconfirmed_save_is_reported_until_a_clean_reply_arrives(persona_dir: Path, monkeypatch):
    from brain.bridge.usage_log import log_usage

    quota = json.dumps({"is_error": True, "api_error_status": 429, "result": "You've hit your session limit"})
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", _Run(lambda m: _proc(1, quota))):
        assert _post_model(c, "claude-sonnet-5").json()["checked"] is False
        assert c.get("/persona/state").json()["model_unconfirmed"] == ["claude-sonnet-5"]
        # a clean reply for ANOTHER model does not clear it; one for this model does
        log_usage(persona_dir, call_type="chat", model="haiku", frame={"is_error": False})
        assert c.get("/persona/state").json()["model_unconfirmed"] == ["claude-sonnet-5"]
        log_usage(persona_dir, call_type="chat", model="claude-sonnet-5", frame={"is_error": False})
        assert c.get("/persona/state").json()["model_unconfirmed"] == []


def test_a_passing_recheck_of_the_same_unconfirmed_model_clears_the_mark(persona_dir: Path, monkeypatch):
    quota = json.dumps({"is_error": True, "api_error_status": 429, "result": "You've hit your session limit"})
    mode = {"ok": False}
    run = _Run(lambda m: _proc(0, _ok_json()) if mode["ok"] else _proc(1, quota))
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        assert _post_model(c, "claude-sonnet-5").json()["checked"] is False
        assert c.get("/persona/state").json()["model_unconfirmed"] == ["claude-sonnet-5"]
        mode["ok"] = True  # the same id again, now the check passes (old == new, so switch() is a no-op)
        assert _post_model(c, "claude-sonnet-5").json()["checked"] is True
        assert c.get("/persona/state").json()["model_unconfirmed"] == []


def test_confirmed_save_and_alias_leave_nothing_unconfirmed(persona_dir: Path, monkeypatch):
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", _Run(lambda m: _proc(0, _ok_json()))):
        _post_model(c, "claude-sonnet-5")
        assert c.get("/persona/state").json()["model_unconfirmed"] == []
        _post_model(c, "opus")
        assert c.get("/persona/state").json()["model_unconfirmed"] == []


def test_auth_expired_saves_with_checked_false_and_makes_no_call(persona_dir: Path, monkeypatch, caplog):
    run = _Run(lambda m: _proc(0, _ok_json()))
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        provider_auth.note_cli_failure(LIVE)  # flips to expired and starts the probe interval
        with caplog.at_level(logging.INFO):
            r = _post_model(c, "claude-sonnet-5")
    # the derived tier id could not be probed either (same failure), so it is reported as not checked yet
    assert r.json() == {"ok": True, "model": "claude-sonnet-5", "checked": False, "tiers_unchecked": ["claude-haiku-5"]}
    assert run.calls_for("claude-sonnet-5") == 0
    assert any("reason=auth_deferred" in rec.getMessage() for rec in caplog.records)


def test_timeout_saves_with_checked_false(persona_dir: Path, monkeypatch, caplog):
    def _boom(model):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", _Run(_boom)):
        with caplog.at_level(logging.INFO):
            r = _post_model(c, "claude-sonnet-5")
    # the derived tier id could not be probed either (same failure), so it is reported as not checked yet
    assert r.json() == {"ok": True, "model": "claude-sonnet-5", "checked": False, "tiers_unchecked": ["claude-haiku-5"]}
    assert any("reason=timeout" in rec.getMessage() for rec in caplog.records)


def test_alias_is_never_checked_and_body_is_unchanged(persona_dir: Path, monkeypatch):
    run = _Run(lambda m: _proc(0, _ok_json()))
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        r = _post_model(c, "opus")
    assert r.json() == {"ok": True, "model": "opus"}
    assert run.calls_for("opus") == 0


def test_non_claude_provider_saves_unchecked(persona_dir: Path):
    run = _Run(lambda m: _proc(0, _ok_json()))
    app = build_app(persona_dir=persona_dir, client_origin="tests")  # persona_config says provider: fake
    with TestClient(app) as c, patch("subprocess.run", run):
        r = _post_model(c, "claude-sonnet-5")
    assert r.json() == {"ok": True, "model": "claude-sonnet-5"}
    assert run.calls_for("claude-sonnet-5") == 0


def test_concurrent_applies_run_their_checks_one_at_a_time(persona_dir: Path, monkeypatch):
    checks = frozenset({"claude-aaa-1", "claude-bbb-2"})
    run = _Run(lambda m: _proc(0, _ok_json()), delay=0.25, watch=checks)
    results: list[int] = []
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        def _go(mid):
            results.append(_post_model(c, mid).status_code)

        ts = [threading.Thread(target=_go, args=(m,)) for m in ("claude-aaa-1", "claude-bbb-2")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert results == [200, 200]
    assert sum(run.calls_for(m) for m in checks) == 2
    assert run.max_inflight == 1  # single-flight: the second check waited for the first


# --- C16: visibility of a saved id that later stops working -------------------------------------------


def _persona_with_model(persona_dir: Path, model: str) -> None:
    (persona_dir / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "fake", "model": model}))


def test_ambient_generate_failure_is_remembered_and_shown_in_persona_state(persona_dir: Path, monkeypatch):
    _persona_with_model(persona_dir, BAD)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")) as c:
        with patch("subprocess.run", return_value=_proc(1, REJECT_RESULT)):
            with pytest.raises(RuntimeError):
                ClaudeCliProvider(model=BAD).generate("tick")
        recs = c.get("/persona/state").json()["model_rejections"]
    assert [r["model"] for r in recs] == [BAD]
    assert recs[0]["count"] == 1


def test_rejection_of_an_unrelated_model_is_not_returned(persona_dir: Path):
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")) as c:
        with patch("subprocess.run", return_value=_proc(1, REJECT_RESULT)):
            with pytest.raises(RuntimeError):
                ClaudeCliProvider(model=BAD).generate("tick")  # configured model is the default, BAD is not it
        assert model_error.state() != []
        assert c.get("/persona/state").json()["model_rejections"] == []


def test_persona_state_has_an_empty_list_when_nothing_was_rejected(persona_dir: Path):
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")) as c:
        st = c.get("/persona/state").json()
    assert st["model_rejections"] == []
    assert "provider_auth_expired" in st


def test_switching_model_forgets_the_old_ones_record(persona_dir: Path, monkeypatch):
    _persona_with_model(persona_dir, BAD)
    with _client(persona_dir, monkeypatch, model=BAD) as c, patch("subprocess.run", _Run(lambda m: _proc(0, _ok_json()))):
        model_error.note_failure(FRAMES[1]["result"])
        assert [r["model"] for r in c.get("/persona/state").json()["model_rejections"]] == [BAD]
        assert _post_model(c, "sonnet").status_code == 200
        assert c.get("/persona/state").json()["model_rejections"] == []
        model_error.note_failure(FRAMES[1]["result"])  # a late in-flight failure for the old id
        assert model_error.state() == []


# --- C18: the committed fixture is scrubbed ---------------------------------------------------------


def test_fixture_is_scrubbed_and_only_holds_assistant_and_result_frames():
    text = FIXTURE.read_text(encoding="utf-8")
    for needle in ("/home/", "session_id", "uuid", "request_id", "memory_paths", getpass.getuser()):
        assert needle not in text, needle
    assert {f.get("type") for f in FRAMES} == {"assistant", "result"}
    assert len(FRAMES) == 4
