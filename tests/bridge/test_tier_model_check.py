"""#441 scope addition: the picker's Apply also checks the tier models derived from the chat id, records
which fell back, and keeps alias users byte-for-byte unchanged. Real ClaudeCliProvider + routes; only the
subprocess is faked (see test_model_unavailable_surface.py)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from brain.bridge import model_error
from brain.bridge.model_tier import (
    TIER_ATTUNEMENT_DETECTOR,
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_COMPACTION,
    build_tier_provider,
    resolve_tier_model,
)
from brain.bridge.provider import ClaudeCliProvider
from brain.bridge.server import build_app
from brain.persona_config import PersonaConfig

PINNED = "claude-haiku-4-5-20251001"
SENTENCE = "There's an issue with the selected model ({model}). It may not exist or you may not have access to it."
QUOTA = json.dumps({"is_error": True, "api_error_status": 429, "result": "You've hit your session limit"})


def _proc(rc: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def _ok() -> subprocess.CompletedProcess:
    return _proc(0, json.dumps({"is_error": False, "result": "OK"}))


def _rejected(model: str) -> subprocess.CompletedProcess:
    return _proc(1, json.dumps({"is_error": True, "api_error_status": 404, "result": SENTENCE.format(model=model)}))


class _Run:
    def __init__(self, respond, delay: float = 0.0, watch=frozenset()):
        self.respond, self.delay, self.watch = respond, delay, watch
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


def _cli_persona(persona_dir: Path, model: str = "sonnet", **extra) -> Path:
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "claude-cli", "searcher": "noop", "model": model, **extra})
    )
    return persona_dir / "persona_config.json"


def _client(persona_dir: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr("brain.bridge.provider.get_provider", lambda *a, **k: ClaudeCliProvider(model="sonnet"))
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


def _post(c: TestClient, model: str):
    return c.post("/persona/config/model", json={"model": model})


def _cfg(persona_dir: Path) -> PersonaConfig:
    return PersonaConfig.load(persona_dir / "persona_config.json")


FALLBACKS = [
    {"family": "haiku", "wanted": "claude-haiku-5", "using": "haiku"},
    {"family": "haiku", "wanted": "claude-haiku-5", "using": PINNED},
]


def test_rejected_derived_id_is_recorded_as_a_fallback(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    run = _Run(lambda m: _rejected(m) if m == "claude-haiku-5" else _ok())
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        r = _post(c, "claude-sonnet-5")
        state = c.get("/persona/state").json()
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "model": "claude-sonnet-5", "checked": True, "fallbacks": FALLBACKS}
    assert _cfg(persona_dir).derived_unavailable == ["claude-haiku-5"]
    assert resolve_tier_model(persona_dir, TIER_COMPACTION) == ("haiku", "fallback")
    assert resolve_tier_model(persona_dir, TIER_ATTUNEMENT_DETECTOR) == (PINNED, "fallback")
    assert resolve_tier_model(persona_dir, TIER_BACKGROUND_GENERATIVE) == ("claude-sonnet-5", "derived")
    assert model_error.state() == []  # the checks are suppressed: nothing recorded for a probe
    assert state["model_rejections"] == []
    assert state["tier_fallbacks"] == FALLBACKS


def test_refused_chat_id_changes_nothing_on_disk(persona_dir: Path, monkeypatch):
    path = _cli_persona(persona_dir, "sonnet", derived_unavailable=["claude-haiku-5"])
    before = path.read_bytes()
    run = _Run(lambda m: _rejected(m) if m == "claude-sonnet-5" else _ok())
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        r = _post(c, "claude-sonnet-5")
    assert r.status_code == 400 and r.json()["error"] == "model_unavailable"
    assert path.read_bytes() == before


def test_alias_apply_makes_no_derived_check_and_writes_no_new_keys(persona_dir: Path, monkeypatch):
    path = _cli_persona(persona_dir, "sonnet")
    run = _Run(lambda m: _ok())
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        r = _post(c, "opus")
    assert r.json() == {"ok": True, "model": "opus"}
    assert run.calls_for("claude-haiku-5") == 0 and run.calls_for("claude-opus-5") == 0
    saved = json.loads(path.read_text())
    assert saved["model"] == "opus"
    assert "tier_models" not in saved and "derived_unavailable" not in saved


def test_alias_apply_clears_a_stale_unavailable_list(persona_dir: Path, monkeypatch):
    path = _cli_persona(persona_dir, "claude-sonnet-5", derived_unavailable=["claude-haiku-5"])
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", _Run(lambda m: _ok())):
        assert _post(c, "sonnet").json() == {"ok": True, "model": "sonnet"}
    assert "derived_unavailable" not in json.loads(path.read_text())


def test_inconclusive_recheck_keeps_an_earlier_rejection_for_the_same_chat_id(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _rejected(m) if m == "claude-haiku-5" else _ok())):
            _post(c, "claude-sonnet-5")
        assert _cfg(persona_dir).derived_unavailable == ["claude-haiku-5"]
        with patch("subprocess.run", _Run(lambda m: _proc(1, QUOTA) if m == "claude-haiku-5" else _ok())):
            r = _post(c, "claude-sonnet-5")  # re-Apply: the derived check is inconclusive now
        assert _cfg(persona_dir).derived_unavailable == ["claude-haiku-5"]
        assert r.json()["fallbacks"] == FALLBACKS
        with patch("subprocess.run", _Run(lambda m: _ok())):
            r = _post(c, "claude-sonnet-5")  # passes now: recovered
        assert _cfg(persona_dir).derived_unavailable == []
        assert "fallbacks" not in r.json()


def test_a_family_whose_tiers_are_all_overridden_is_not_checked(persona_dir: Path, monkeypatch):
    overrides = {t: "haiku" for t in ("attunement-detector", "compaction", "self-model-articulate", "background-classifier", "background-housekeeping")}
    _cli_persona(persona_dir, tier_models=overrides)
    run = _Run(lambda m: _ok())
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        _post(c, "claude-sonnet-5")
    assert run.calls_for("claude-haiku-5") == 0


def test_chat_and_derived_checks_run_together_and_applies_still_serialize(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    watch = frozenset({"claude-sonnet-5", "claude-haiku-5", "claude-opus-4-8", "claude-haiku-4-8"})
    run = _Run(lambda m: _ok(), delay=0.25, watch=watch)
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        _post(c, "claude-sonnet-5")
        assert run.max_inflight == 2  # chat id and the derived haiku id at the same time
        run.max_inflight = 0
        out: list[int] = []
        ts = [threading.Thread(target=lambda m=m: out.append(_post(c, m).status_code)) for m in ("claude-opus-4-8", "claude-sonnet-5")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert out == [200, 200]
    assert run.max_inflight <= 2  # never four: the second Apply waited for the first


def test_switch_away_and_back_as_a_derived_id_still_falls_back_when_rejected(persona_dir: Path, monkeypatch):
    # M1 through the real endpoint: claude-sonnet-5 is ignored after switching to an alias, then becomes the
    # sonnet-family tier's derived id when the chat model is claude-opus-5.
    _cli_persona(persona_dir, "claude-sonnet-5")
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _ok())):
            _post(c, "sonnet")
            _post(c, "claude-opus-5")
        assert resolve_tier_model(persona_dir, TIER_BACKGROUND_GENERATIVE) == ("claude-sonnet-5", "derived")
        with patch("subprocess.run", _Run(lambda m: _rejected(m) if m == "claude-sonnet-5" else _ok())):
            try:
                build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE).generate("tick")
            except RuntimeError:
                pass
        assert resolve_tier_model(persona_dir, TIER_BACKGROUND_GENERATIVE) == ("sonnet", "fallback")
        assert "claude-sonnet-5" in {r["model"] for r in c.get("/persona/state").json()["model_rejections"]}


def test_runtime_fallback_is_reported_in_persona_state(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir, "claude-sonnet-5")
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _rejected(m) if m == "claude-haiku-5" else _ok())):
            try:
                build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER).generate("tick")
            except RuntimeError:
                pass
        st = c.get("/persona/state").json()
    assert st["tier_fallbacks"] == FALLBACKS
    assert "claude-haiku-5" in {r["model"] for r in st["model_rejections"]}  # still reported while replaced


def test_chat_model_rejection_is_reported_and_has_no_fallback_entry(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir, "claude-sonnet-5")
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _rejected(m) if m == "claude-sonnet-5" else _ok())):
            try:
                build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE).generate("tick")
            except RuntimeError:
                pass
        st = c.get("/persona/state").json()
    assert "claude-sonnet-5" in {r["model"] for r in st["model_rejections"]}
    assert st["tier_fallbacks"] == []
