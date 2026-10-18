"""#449 - the saved ``derived_unavailable`` ids are tested again when the bridge starts.

Real lifespan (background threads on, with the supervisor and the other start-up threads stubbed), real
ClaudeCliProvider and routes; only the subprocess is faked. The re-check shares the picker Apply's probe helper,
budget and single-flight lock, runs off the start-up critical path, and writes with ONE synchronous
load -> modify -> save step (no ``await`` between), skipping the write when the chat model changed meanwhile.
"""

from __future__ import annotations

import inspect
import json
import logging
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain.bridge import model_error, provider_auth
from brain.bridge import server as server_mod
from brain.bridge.model_tier import TIER_BACKGROUND_CLASSIFIER, resolve_tier_model
from brain.bridge.provider import ClaudeCliProvider
from brain.bridge.server import build_app
from brain.persona_config import PersonaConfig

CHAT = "claude-sonnet-5"
DERIVED = "claude-haiku-5"
SENT = "There's an issue with the selected model ({model}). It may not exist or you may not have access to it."
AUTH = "Failed to authenticate: OAuth session expired and could not be refreshed"


@pytest.fixture(autouse=True)
def _reset():
    model_error.reset()
    yield
    model_error.reset()


@pytest.fixture
def bg_stubs(monkeypatch):
    """Background threads ON (the re-check is gated on it) but the heavy ones replaced by no-ops."""
    import brain.bridge.supervisor as sup
    import brain.chat.compaction_migration as cm
    import brain.memory.floor_startup as fs

    monkeypatch.setattr(sup, "run_folded", lambda *, stop_event, **kw: stop_event.wait())
    monkeypatch.setattr(fs, "start_background", lambda *a, **k: None)
    monkeypatch.setattr(cm, "run_backlog_migration", lambda *a, **k: None)
    monkeypatch.setattr(cm, "run_sections_migration", lambda *a, **k: None)


def _ok_json() -> str:
    return json.dumps({"is_error": False, "result": "OK"})


def _rej_json(model: str) -> str:
    return json.dumps({"is_error": True, "api_error_status": 404, "result": SENT.format(model=model)})


def _proc(rc: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


class _Run:
    """Recording stand-in for subprocess.run: ``respond(model) -> CompletedProcess`` (may raise)."""

    def __init__(self, respond, delay: float = 0.0):
        self.respond = respond
        self.delay = delay
        self.models: list[str] = []
        self._lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    def __call__(self, cmd, **kw):
        model = cmd[cmd.index("--model") + 1] if "--model" in cmd else ""
        with self._lock:
            self.models.append(model)
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self.respond(model)
        finally:
            with self._lock:
                self.inflight -= 1

    def calls_for(self, model: str) -> int:
        return self.models.count(model)


def _write(persona_dir: Path, model: str = CHAT, unavailable=(DERIVED,), provider: str = "claude-cli", **extra) -> Path:
    p = persona_dir / "persona_config.json"
    data = {"provider": provider, "searcher": "noop", "model": model, "user_name": "Bob", **extra}
    if unavailable:
        data["derived_unavailable"] = list(unavailable)
    p.write_text(json.dumps(data))
    return p


def _app(persona_dir: Path, monkeypatch, *, bg: bool = True, provider_model: str = CHAT, kind=ClaudeCliProvider):
    monkeypatch.setattr("brain.bridge.provider.get_provider", lambda *a, **k: kind(model=provider_model))
    return build_app(persona_dir=persona_dir, client_origin="tests", background_threads=bg)


def _wait(cond, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def _recheck_done(app) -> bool:
    task = getattr(app.state, "derived_recheck_task", None)
    return task is not None and task.done()


def _list(p: Path) -> list[str]:
    return PersonaConfig.load(p).derived_unavailable


# --- MC1: outcomes -------------------------------------------------------------------------------------------


def test_available_again_is_removed_and_the_tier_follows_the_derived_id(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir)
    model_error.note_failure(SENT.format(model=DERIVED))  # an earlier runtime record for the same id
    assert [r["model"] for r in model_error.state()] == [DERIVED]
    run = _Run(lambda m: _proc(0, _ok_json()))
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
        assert _list(p) == []
        assert resolve_tier_model(persona_dir, TIER_BACKGROUND_CLASSIFIER) == (DERIVED, "derived")
        state = c.get("/persona/state").json()
        assert state["tier_fallbacks"] == [] and state["model_rejections"] == []
    saved = json.loads(p.read_text())
    assert "derived_unavailable" not in saved
    assert saved["user_name"] == "Bob" and saved["model"] == CHAT
    assert run.calls_for(DERIVED) == 1 and run.calls_for(CHAT) == 0


def test_still_rejected_leaves_the_list_and_does_not_rewrite_the_file(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir)
    saves: list[str] = []
    real_save = PersonaConfig.save
    monkeypatch.setattr(PersonaConfig, "save", lambda self, path: (saves.append(str(path)), real_save(self, path))[1])
    run = _Run(lambda m: _proc(1, _rej_json(m)))
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
        assert _list(p) == [DERIVED]
        assert resolve_tier_model(persona_dir, TIER_BACKGROUND_CLASSIFIER) == ("haiku", "fallback")
    assert len(saves) == 1, saves  # only the lifespan's own last_opened_at touch; the re-check wrote nothing
    assert run.calls_for(DERIVED) == 1


@pytest.mark.parametrize("kind", ["quota", "timeout", "auth"])
def test_inconclusive_leaves_the_list(persona_dir: Path, monkeypatch, bg_stubs, kind):
    p = _write(persona_dir)

    def respond(model):
        if kind == "quota":
            return _proc(1, json.dumps({"is_error": True, "api_error_status": 429, "result": "You've hit your session limit"}))
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    run = _Run(respond)
    if kind == "auth":
        provider_auth.note_cli_failure(AUTH)  # flips to expired: generate defers without a call
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
        assert _list(p) == [DERIVED]
    if kind == "auth":
        assert run.calls_for(DERIVED) == 0


def test_only_the_ids_that_passed_are_removed(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir, unavailable=(DERIVED, "claude-haiku-6"))
    run = _Run(lambda m: _proc(0, _ok_json()) if m == DERIVED else _proc(1, _rej_json(m)))
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
    assert _list(p) == ["claude-haiku-6"]


@pytest.mark.parametrize(
    "case",
    ["empty_list", "alias_chat", "fake_provider", "bg_off"],
)
def test_no_cli_call_when_there_is_nothing_to_recheck(persona_dir: Path, monkeypatch, bg_stubs, case):
    kwargs = {}
    if case == "empty_list":
        _write(persona_dir, unavailable=())
    elif case == "alias_chat":
        _write(persona_dir, model="opus")
    elif case == "fake_provider":
        _write(persona_dir)
        kwargs["kind"] = lambda model: __import__("brain.bridge.provider", fromlist=["x"]).FakeProvider()
    else:
        _write(persona_dir)
        kwargs["bg"] = False
    run = _Run(lambda m: _proc(0, _ok_json()))
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch, **kwargs)) as c:
        time.sleep(0.3)
        assert getattr(c.app.state, "derived_recheck_task", None) is None
    assert run.models == []


# --- MC2: off the critical path ----------------------------------------------------------------------------------


def test_bridge_is_ready_and_answers_while_the_recheck_probe_is_still_blocked(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir)
    entered, release, started = threading.Event(), threading.Event(), threading.Event()

    def respond(model):
        entered.set()
        release.wait(20)
        return _proc(0, _ok_json())

    holder: dict = {}
    client = TestClient(_app(persona_dir, monkeypatch))

    def enter():
        holder["c"] = client.__enter__()  # runs the lifespan start-up
        started.set()

    with patch("subprocess.run", _Run(respond)):
        t = threading.Thread(target=enter, daemon=True)
        t.start()
        try:
            assert entered.wait(10), "the re-check never started its probe"
            # hard timeout: a start-up that awaits the re-check before readiness stays blocked here
            assert started.wait(5), "start-up waited for the re-check (not off the critical path)"
            answered: dict = {}
            h = threading.Thread(target=lambda: answered.update(status=holder["c"].get("/health").status_code), daemon=True)
            h.start()
            h.join(5)
            assert not h.is_alive() and "status" in answered, "the bridge did not answer while the probe was blocked"
        finally:
            release.set()
            t.join(15)
            if started.is_set():
                _wait(lambda: _recheck_done(client.app))
                client.__exit__(None, None, None)
        assert _list(p) == []


# --- MC3: same bounded, single-flight check -------------------------------------------------------------------------


def test_budget_used_up_makes_probes_inconclusive_without_calls(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir)
    monkeypatch.setattr(server_mod, "_check_budget_s", lambda: 0.5)  # below _MIN_PROBE_S
    run = _Run(lambda m: _proc(0, _ok_json()))
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
    assert run.models == [] and _list(p) == [DERIVED]


def test_saved_ids_are_probed_together(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir, unavailable=(DERIVED, "claude-haiku-6", "claude-opus-5"))
    run = _Run(lambda m: _proc(0, _ok_json()), delay=0.3)
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
    assert run.max_inflight > 1 and _list(p) == []


def test_an_apply_during_the_recheck_waits_for_it_and_never_overlaps(persona_dir: Path, monkeypatch, bg_stubs):
    _write(persona_dir)
    entered, release = threading.Event(), threading.Event()
    skew = {"v": 0.0}
    real = time.monotonic
    monkeypatch.setattr(server_mod, "_clock", lambda: real() + skew["v"])

    def respond(model):
        if model == DERIVED and not entered.is_set():
            entered.set()
            release.wait(20)
            skew["v"] += 29.5  # the re-check used almost the whole 30 s budget while the Apply waited
        return _proc(0, _ok_json())

    run = _Run(respond)
    out: dict = {}
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert entered.wait(10)

        def apply():
            out["r"] = c.post("/persona/config/model", json={"model": "claude-opus-5"})

        t = threading.Thread(target=apply, daemon=True)
        t.start()
        time.sleep(0.3)
        assert t.is_alive(), "the Apply must wait for the re-check (single-flight)"
        release.set()
        t.join(15)
        assert not t.is_alive()
    body = out["r"].json()
    assert out["r"].status_code == 200 and body["checked"] is False  # budget gone: saved unchecked, did not hang
    assert run.max_inflight == 1  # never two generate calls at once, across the re-check and the Apply
    assert run.calls_for("claude-opus-5") == 0


# --- MC4: no lost update ------------------------------------------------------------------------------------------------


def test_apply_equivalent_write_to_another_chat_model_during_the_probe_is_not_clobbered(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir)
    written: dict = {}

    def respond(model):
        cfg = PersonaConfig.load(p)
        cfg.set_model("opus")  # what an alias Apply does (it needs no lock, so it can land mid-probe)
        cfg.derived_unavailable = []
        cfg.save(p)
        written["bytes"] = p.read_bytes()
        return _proc(0, _ok_json())

    with patch("subprocess.run", _Run(respond)), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
    final = json.loads(p.read_bytes())
    assert final["model"] == "opus"
    assert p.read_bytes() == written["bytes"], "the re-check wrote over a newer config"


def test_a_competing_write_of_another_field_during_the_probe_survives(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir)

    def respond(model):
        cfg = PersonaConfig.load(p)
        replace(cfg, user_name="Injected").save(p)
        return _proc(0, _ok_json())

    with patch("subprocess.run", _Run(respond)), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
    final = json.loads(p.read_text())
    assert final["user_name"] == "Injected" and "derived_unavailable" not in final


def test_the_update_step_is_synchronous_and_loads_then_saves_adjacent(persona_dir: Path, monkeypatch):
    assert not inspect.iscoroutinefunction(server_mod._apply_recheck_outcomes)
    p = _write(persona_dir)
    calls: list[str] = []
    real_load, real_save = PersonaConfig.load.__func__, PersonaConfig.save
    monkeypatch.setattr(PersonaConfig, "load", classmethod(lambda cls, path: (calls.append("load"), real_load(cls, path))[1]))
    monkeypatch.setattr(PersonaConfig, "save", lambda self, path: (calls.append("save"), real_save(self, path))[1])
    removed = server_mod._apply_recheck_outcomes(persona_dir, CHAT, {DERIVED: "ok"})
    assert removed == [DERIVED] and calls == ["load", "save"]
    assert _list(p) == []


def test_update_step_skips_when_the_chat_model_changed_or_nothing_passed(persona_dir: Path):
    p = _write(persona_dir)
    before = p.read_bytes()
    assert server_mod._apply_recheck_outcomes(persona_dir, "claude-opus-5", {DERIVED: "ok"}) == []
    assert server_mod._apply_recheck_outcomes(persona_dir, CHAT, {DERIVED: "rejected"}) == []
    assert server_mod._apply_recheck_outcomes(persona_dir, CHAT, {DERIVED: "inconclusive"}) == []
    assert p.read_bytes() == before


# --- MC5: shutdown -----------------------------------------------------------------------------------------------------------


def test_shutdown_during_a_probe_cancels_cleanly(persona_dir: Path, monkeypatch, bg_stubs, caplog):
    _write(persona_dir)
    entered, release = threading.Event(), threading.Event()

    def respond(model):
        entered.set()
        release.wait(20)
        return _proc(0, _ok_json())

    with caplog.at_level(logging.ERROR), patch("subprocess.run", _Run(respond)):
        with TestClient(_app(persona_dir, monkeypatch)):
            assert entered.wait(10)
            threading.Timer(0.5, release.set).start()  # frees the probe thread so the executor can stop
        release.set()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR and "recheck" in r.getMessage().lower()]


# --- MC6 / MC7 / MC8 -------------------------------------------------------------------------------------------------------------


def test_nell_model_show_note_says_both_when_the_list_is_tested(monkeypatch, tmp_path: Path, capsys):
    from brain import cli

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    d = tmp_path / "personas" / "bob"
    d.mkdir(parents=True)
    _write(d)
    cli._build_parser()
    with patch("subprocess.run", side_effect=AssertionError("no CLI call")):
        assert cli.main(["model", "show", "--persona", "bob"]) == 0
    out = capsys.readouterr().out
    assert "when the bridge starts" in out and "applied again" in out


def test_each_rechecked_id_is_logged_with_its_outcome(persona_dir: Path, monkeypatch, bg_stubs, caplog):
    _write(persona_dir, unavailable=(DERIVED, "claude-haiku-6"))
    run = _Run(lambda m: _proc(0, _ok_json()) if m == DERIVED else _proc(1, _rej_json(m)))
    with caplog.at_level(logging.INFO), patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
    msgs = [r.getMessage() for r in caplog.records if r.name.endswith("server")]
    assert any(DERIVED in m and "ok" in m for m in msgs)
    assert any("claude-haiku-6" in m and "rejected" in m for m in msgs)
    assert not any(persona_dir.name in m for m in msgs if "recheck" in m.lower())


# --- restart throttle (no probe storm on a crash loop) -----------------------------------------------------------------


def test_a_second_start_within_the_reprobe_interval_does_not_probe_again(persona_dir: Path, monkeypatch, bg_stubs):
    p = _write(persona_dir)
    run = _Run(lambda m: _proc(1, _rej_json(m)))  # still rejected: the id stays saved, so a restart would re-probe
    with patch("subprocess.run", run):
        with TestClient(_app(persona_dir, monkeypatch)) as c:
            assert _wait(lambda: _recheck_done(c.app))
        assert run.calls_for(DERIVED) == 1
        assert (persona_dir / server_mod._RECHECK_STAMP).exists()
        for _ in range(3):  # a crash loop
            with TestClient(_app(persona_dir, monkeypatch)) as c:
                time.sleep(0.2)
                assert getattr(c.app.state, "derived_recheck_task", None) is None
    assert run.calls_for(DERIVED) == 1 and _list(p) == [DERIVED]


def test_an_old_stamp_lets_the_recheck_run_again(persona_dir: Path, monkeypatch, bg_stubs):
    import os

    p = _write(persona_dir)
    stamp = persona_dir / server_mod._RECHECK_STAMP
    stamp.write_text("")
    old = time.time() - provider_auth._reprobe_s() - 5
    os.utime(stamp, (old, old))
    run = _Run(lambda m: _proc(0, _ok_json()))
    with patch("subprocess.run", run), TestClient(_app(persona_dir, monkeypatch)) as c:
        assert _wait(lambda: _recheck_done(c.app))
    assert run.calls_for(DERIVED) == 1 and _list(p) == []
    assert time.time() - stamp.stat().st_mtime < 60  # restamped


def test_no_stamp_is_written_when_there_is_nothing_to_recheck(persona_dir: Path, monkeypatch, bg_stubs):
    _write(persona_dir, unavailable=())
    with patch("subprocess.run", _Run(lambda m: _proc(0, _ok_json()))), TestClient(_app(persona_dir, monkeypatch)):
        time.sleep(0.2)
    assert not (persona_dir / server_mod._RECHECK_STAMP).exists()
