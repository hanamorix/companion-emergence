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
    # honour model_override like the real get_provider, so tier providers carry their resolved model
    monkeypatch.setattr(
        "brain.bridge.provider.get_provider",
        lambda name="claude-cli", persona_dir=None, model_override=None, **k: ClaudeCliProvider(model=model_override or "sonnet"),
    )
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
    run = _Run(lambda m: _rejected(m) if m == "claude-sonnet-5" else _ok())
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        before = path.read_bytes()  # after the app's own start-up write (last_opened_at)
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
    overrides = dict.fromkeys(("attunement-detector", "compaction", "self-model-articulate", "background-classifier", "background-housekeeping"), "haiku")
    _cli_persona(persona_dir, tier_models=overrides)
    run = _Run(lambda m: _ok())
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        _post(c, "claude-sonnet-5")
    assert run.calls_for("claude-haiku-5") == 0


def test_derived_checks_run_together_after_the_chat_check_and_applies_still_serialize(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    # chat claude-opus-4-8 derives two ids (haiku and sonnet family): they are probed at the same time,
    # but only after the chat id itself was probed
    watch = frozenset({"claude-opus-4-8", "claude-haiku-4-8", "claude-sonnet-4-8"})
    order: list[str] = []
    run = _Run(lambda m: order.append(m) or _ok(), delay=0.25, watch=watch)
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        assert _post(c, "claude-opus-4-8").status_code == 200
        assert order[0] == "claude-opus-4-8" and set(order[1:3]) == {"claude-haiku-4-8", "claude-sonnet-4-8"}
        assert run.max_inflight == 2  # the two derived probes overlap; the chat probe ran alone first
        run.max_inflight = 0
        out: list[int] = []
        ts = [threading.Thread(target=lambda m=m: out.append(_post(c, m).status_code)) for m in ("claude-opus-4-8", "claude-sonnet-5")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert out == [200, 200]
    assert run.max_inflight <= 2  # never overlapping across the two Applies: the second waited for the first


def test_no_derived_probe_is_spent_when_the_chat_id_is_rejected(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    run = _Run(lambda m: _rejected(m) if m == "claude-sonet-5" else _ok())
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        r = _post(c, "claude-sonet-5")
    assert r.status_code == 400
    assert run.calls_for("claude-sonet-5") == 1
    assert run.calls_for("claude-haiku-5") == 0  # the derived id of a sonnet-family chat id


def test_a_passing_recheck_clears_a_runtime_rejection_and_the_fallback(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir, "claude-sonnet-5")
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _rejected(m) if m == "claude-haiku-5" else _ok())):
            try:
                build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER).generate("tick")
            except RuntimeError:
                pass
        assert resolve_tier_model(persona_dir, TIER_COMPACTION) == ("haiku", "fallback")
        with patch("subprocess.run", _Run(lambda m: _ok())):  # the account gained access
            r = _post(c, "claude-sonnet-5")
        body = r.json()
        st = c.get("/persona/state").json()
    assert "fallbacks" not in body and body["checked"] is True
    assert resolve_tier_model(persona_dir, TIER_COMPACTION) == ("claude-haiku-5", "derived")
    assert st["tier_fallbacks"] == [] and st["model_rejections"] == []


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


# --- one shared time budget for all the probes of one Apply -----------------------------------------


class _TimedRun:
    """subprocess.run stand-in with per-model durations that honours the `timeout=` it is given, like the real one."""

    def __init__(self, durations: dict[str, float]):
        self.durations = durations
        self.timeouts: dict[str, int] = {}
        self.calls: list[str] = []

    def __call__(self, cmd, **kw):
        model = cmd[cmd.index("--model") + 1]
        self.calls.append(model)
        timeout = kw.get("timeout")
        self.timeouts.setdefault(model, timeout)
        took = self.durations.get(model, 0.0)
        if timeout is not None and took > timeout:
            time.sleep(timeout)
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
        time.sleep(took)
        return _ok()


def test_all_probes_of_one_apply_share_one_budget(persona_dir: Path, monkeypatch):
    # The budget is the whole Apply, not each probe: otherwise a hanging CLI would hold the request for
    # chat + derived time, past a webview's ~60 s idle limit. Here: budget 3 s, chat probe 1.2 s, derived hang.
    from brain.bridge import server as server_mod

    monkeypatch.setattr(server_mod, "_MODEL_CHECK_TIMEOUT_S", 3.0)
    _cli_persona(persona_dir)
    run = _TimedRun({"claude-sonnet-5": 1.2, "claude-haiku-5": 100.0})
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        t0 = time.monotonic()
        r = _post(c, "claude-sonnet-5")
        elapsed = time.monotonic() - t0
    assert r.status_code == 200 and r.json()["checked"] is True
    assert run.timeouts["claude-sonnet-5"] == 3  # the chat probe may use the whole budget
    assert run.timeouts["claude-haiku-5"] <= 2  # derived probes only get what is left
    assert elapsed < 3.0 + 1.0  # never chat + derived: the total stays inside the budget (plus margin)
    assert _cfg(persona_dir).derived_unavailable == []  # a probe that timed out says nothing: not recorded


def test_derived_probes_are_skipped_when_the_chat_probe_used_the_budget(persona_dir: Path, monkeypatch):
    from brain.bridge import server as server_mod
    from brain.bridge.usage_log import log_usage

    monkeypatch.setattr(server_mod, "_MODEL_CHECK_TIMEOUT_S", 2.0)
    _cli_persona(persona_dir)
    run = _TimedRun({"claude-sonnet-5": 100.0})  # the chat probe hangs for the whole budget
    with _client(persona_dir, monkeypatch) as c, patch("subprocess.run", run):
        t0 = time.monotonic()
        r = _post(c, "claude-sonnet-5")
        elapsed = time.monotonic() - t0
        st = c.get("/persona/state").json()
        # the user is told, both in the response and in the state the panel polls
        assert r.json()["checked"] is False
        assert r.json()["tiers_unchecked"] == ["claude-haiku-5"]
        assert st["tier_unconfirmed"] == ["claude-haiku-5"]
        # a clean reply for that tier id clears it, like the chat model's own notice
        log_usage(persona_dir, call_type="generate", model="claude-haiku-5", frame={"is_error": False})
        assert c.get("/persona/state").json()["tier_unconfirmed"] == []
    assert r.status_code == 200
    assert "claude-haiku-5" not in run.calls  # no budget left: no derived probe was started
    assert elapsed < 2.0 + 1.0


def test_a_later_apply_whose_probes_all_pass_leaves_nothing_unchecked(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _proc(1, QUOTA) if m == "claude-haiku-5" else _ok())):
            first = _post(c, "claude-sonnet-5").json()
        assert first["tiers_unchecked"] == ["claude-haiku-5"]
        assert c.get("/persona/state").json()["tier_unconfirmed"] == ["claude-haiku-5"]
        with patch("subprocess.run", _Run(lambda m: _ok())):
            second = _post(c, "claude-sonnet-5").json()
        assert "tiers_unchecked" not in second
        assert c.get("/persona/state").json()["tier_unconfirmed"] == []


def test_rejected_derived_ids_and_alias_applies_report_nothing_unchecked(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _rejected(m) if m == "claude-haiku-5" else _ok())):
            body = _post(c, "claude-sonnet-5").json()
        assert "tiers_unchecked" not in body and body["fallbacks"]  # rejected: a fallback, not "unchecked"
        with patch("subprocess.run", _Run(lambda m: _ok())):
            alias = _post(c, "opus").json()
        assert alias == {"ok": True, "model": "opus"}
        assert c.get("/persona/state").json()["tier_unconfirmed"] == []


def test_default_budget_is_well_inside_a_60_second_webview_idle_timeout():
    from brain.bridge import server as server_mod

    assert server_mod._MODEL_CHECK_TIMEOUT_S <= 30.0


def test_an_id_is_never_both_a_fallback_and_not_checked_yet(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir)
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _rejected(m) if m == "claude-haiku-5" else _ok())):
            _post(c, "claude-sonnet-5")  # claude-haiku-5 is rejected: recorded as a fallback
        # re-Apply: the haiku probe is rate limited now; the earlier rejection is kept
        with patch("subprocess.run", _Run(lambda m: _proc(1, QUOTA) if m == "claude-haiku-5" else _ok())):
            body = _post(c, "claude-sonnet-5").json()
        st = c.get("/persona/state").json()
    assert "tiers_unchecked" not in body
    assert {f["wanted"] for f in st["tier_fallbacks"]} == {"claude-haiku-5"}
    assert st["tier_unconfirmed"] == []  # not also "not checked yet"


def test_persona_state_filters_an_id_that_is_a_fallback_even_if_it_was_marked(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir, "claude-sonnet-5", derived_unavailable=["claude-haiku-5"])
    with _client(persona_dir, monkeypatch) as c:
        model_error.note_tier_unconfirmed("claude-haiku-5")  # a stale mark from an earlier Apply
        st = c.get("/persona/state").json()
    assert st["tier_fallbacks"] and st["tier_unconfirmed"] == []


def test_response_never_lists_a_runtime_rejected_id_as_not_checked_yet(persona_dir: Path, monkeypatch):
    _cli_persona(persona_dir, "claude-sonnet-5")
    model_error.note_failure(SENTENCE.format(model="claude-haiku-5"))  # a runtime rejection record (a fallback)
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _proc(1, QUOTA) if m == "claude-haiku-5" else _ok())):
            body = _post(c, "claude-sonnet-5").json()
        st = c.get("/persona/state").json()
    assert {f["wanted"] for f in body["fallbacks"]} == {"claude-haiku-5"}
    assert "tiers_unchecked" not in body  # the same helper filters the response and the state
    assert st["tier_unconfirmed"] == []


def test_a_mark_set_in_an_apply_survives_that_applys_own_switch(persona_dir: Path, monkeypatch):
    # chat model claude-haiku-5 -> claude-sonnet-5: the old chat id is also the new haiku-family derived id
    _cli_persona(persona_dir, "claude-haiku-5")
    with _client(persona_dir, monkeypatch) as c:
        with patch("subprocess.run", _Run(lambda m: _proc(1, QUOTA) if m == "claude-haiku-5" else _ok())):
            body = _post(c, "claude-sonnet-5").json()
        st = c.get("/persona/state").json()
    assert body["tiers_unchecked"] == ["claude-haiku-5"]
    assert st["tier_unconfirmed"] == ["claude-haiku-5"]  # not wiped by the switch that ran after the mark
