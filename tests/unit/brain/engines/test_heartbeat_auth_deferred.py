"""#246 — while the login is expired the heartbeat tick still persists and stays quiet."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.provider_auth import ProviderAuthDeferred
from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


class _DeferringProvider:
    """Stands in for ClaudeCliProvider.generate() while provider_auth is expired."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None, **_: object) -> str:
        self.calls += 1
        raise ProviderAuthDeferred("provider auth expired — deferred")

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)

    def name(self) -> str:
        return "deferring"

    def healthy(self) -> bool:
        return True


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now_value = start

    def now(self, tz=None):
        return self.now_value


def _fake_datetime(clock: _Clock):
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102
            return clock.now_value

    return _DT


@pytest.fixture(autouse=True)
def _no_real_tier_providers(monkeypatch):
    """dream/emit/reflex build their own tier providers; route them to the test's fake."""
    holder = {"provider": _DeferringProvider()}
    monkeypatch.setattr("brain.engines.heartbeat.build_tier_provider", lambda *a, **k: holder["provider"])
    return holder


def _engine(tmp_path: Path, provider, *, reflex_arcs: Path | None = None, research: bool = False) -> tuple[HeartbeatEngine, MemoryStore, HebbianMatrix]:
    store = MemoryStore(tmp_path / "memories.db")
    hm = HebbianMatrix(":memory:")
    store.create(Memory.create_new(content="a seed", memory_type="conversation", domain="us", emotions={"love": 8.0}))
    kwargs = {}
    if reflex_arcs is not None:
        from tests.unit.brain.engines.test_heartbeat import DEFAULT_REFLEX_ARCS_PATH

        kwargs = {
            "reflex_arcs_path": reflex_arcs,
            "reflex_log_path": tmp_path / "reflex_log.json",
            "reflex_default_arcs_path": DEFAULT_REFLEX_ARCS_PATH,
        }
    if research:
        interests = tmp_path / "interests.json"
        interests.write_text(json.dumps({"interests": []}))
        kwargs.update({"interests_path": interests, "research_log_path": tmp_path / "research.log.jsonl"})
    eng = HeartbeatEngine(
        store=store,
        hebbian=hm,
        provider=provider,
        state_path=tmp_path / "heartbeat_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
        **kwargs,
    )
    return eng, store, hm


# C15 — the live scenario: dream 25 h overdue, provider deferring
def test_run_tick_persists_when_overdue_dream_is_deferred(tmp_path: Path, monkeypatch, _no_real_tier_providers) -> None:
    t0 = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    clock = _Clock(t0)
    monkeypatch.setattr("brain.engines.heartbeat.datetime", _fake_datetime(clock))
    HeartbeatConfig(dream_every_hours=24, emit_memory="conditional").save(tmp_path / "hb_config.json")
    st = HeartbeatState.fresh("manual")
    st.last_tick_at = t0 - timedelta(minutes=15)
    st.last_dream_at = t0 - timedelta(hours=25)
    st.save(tmp_path / "heartbeat_state.json")
    provider = _no_real_tier_providers["provider"]
    eng, store, hm = _engine(tmp_path, provider)
    seen: list[float] = []
    real = eng._apply_emotion_decay

    def spy(elapsed_seconds, *, dry_run):
        seen.append(elapsed_seconds)
        return real(elapsed_seconds, dry_run=dry_run)

    monkeypatch.setattr(eng, "_apply_emotion_decay", spy)
    try:
        eng.run_tick(trigger="manual", dry_run=False)
        s1 = HeartbeatState.load(tmp_path / "heartbeat_state.json")
        assert s1.last_tick_at == t0, "first deferred tick must still persist last_tick_at"
        clock.now_value = t0 + timedelta(minutes=15)
        eng.run_tick(trigger="manual", dry_run=False)
        s2 = HeartbeatState.load(tmp_path / "heartbeat_state.json")
        assert s2.last_tick_at == t0 + timedelta(minutes=15)
        assert [round(x) for x in seen] == [900, 900], "decay must be per-interval, not cumulative"
        assert provider.calls >= 1
    finally:
        store.close()
        hm.close()


def test_run_tick_persists_when_emit_is_deferred(tmp_path: Path, monkeypatch, _no_real_tier_providers) -> None:
    t0 = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    clock = _Clock(t0)
    monkeypatch.setattr("brain.engines.heartbeat.datetime", _fake_datetime(clock))
    HeartbeatConfig(dream_every_hours=999, emit_memory="always").save(tmp_path / "hb_config.json")
    st = HeartbeatState.fresh("manual")
    st.last_tick_at = t0 - timedelta(minutes=15)
    st.save(tmp_path / "heartbeat_state.json")
    provider = _no_real_tier_providers["provider"]
    eng, store, hm = _engine(tmp_path, provider)
    try:
        eng.run_tick(trigger="manual", dry_run=False)
        assert HeartbeatState.load(tmp_path / "heartbeat_state.json").last_tick_at == t0
        assert provider.calls == 1
    finally:
        store.close()
        hm.close()


# C17 — reflex with arcs enabled: INFO only, no reflex_error
def test_reflex_deferral_is_quiet(tmp_path: Path, caplog) -> None:
    arcs = tmp_path / "arcs.json"
    arc = {
        "name": "test_arc", "description": "d", "trigger": {"love": 5}, "days_since_human_min": 0,
        "cooldown_hours": 1.0, "action": "a", "output_memory_type": "reflex_journal",
        "prompt_template": "Hi {persona_name}.",
    }
    arcs.write_text(json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8")
    HeartbeatConfig(dream_every_hours=999, emit_memory="conditional", reflex_enabled=True).save(tmp_path / "hb_config.json")
    HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")
    eng, store, hm = _engine(tmp_path, _DeferringProvider(), reflex_arcs=arcs)
    try:
        with caplog.at_level(logging.INFO, logger="brain.engines.heartbeat"):
            result = eng.run_tick(trigger="manual", dry_run=False)
    finally:
        store.close()
        hm.close()
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING and "reflex" in r.getMessage()]
    assert warns == [], [r.getMessage() for r in warns]
    assert result.reflex_error is None


def test_research_deferral_is_quiet(tmp_path: Path, caplog, monkeypatch) -> None:
    HeartbeatConfig(dream_every_hours=999, emit_memory="conditional").save(tmp_path / "hb_config.json")
    HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")
    eng, store, hm = _engine(tmp_path, _DeferringProvider(), research=True)

    def boom(self, *a, **k):
        raise ProviderAuthDeferred("deferred")

    monkeypatch.setattr("brain.engines.research.ResearchEngine.run_tick", boom)
    try:
        with caplog.at_level(logging.INFO, logger="brain.engines.heartbeat"):
            result = eng.run_tick(trigger="manual", dry_run=False)
    finally:
        store.close()
        hm.close()
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING and "research" in r.getMessage()]
    assert warns == []
    assert result.research_gated_reason == "auth_deferred"


def test_voice_reflection_deferral_is_quiet(tmp_path: Path, caplog) -> None:
    from brain.initiate.voice_reflection import run_voice_reflection_tick

    with caplog.at_level(logging.INFO, logger="brain.initiate.voice_reflection"):
        run_voice_reflection_tick(tmp_path, provider=_DeferringProvider(), crystallizations=[], dreams=[])
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# belt: an escaped deferral is visible at WARNING, no traceback
def test_supervisor_belt_logs_warning_without_traceback(tmp_path: Path, caplog, monkeypatch) -> None:
    from brain.bridge import supervisor

    def boom(*a, **k):
        raise ProviderAuthDeferred("escaped")

    monkeypatch.setattr(supervisor, "_run_heartbeat_tick", boom)
    monkeypatch.setattr(supervisor, "_run_felt_time_tick", lambda *a, **k: None)
    with caplog.at_level(logging.INFO, logger="brain.bridge.supervisor"):
        supervisor._heartbeat_and_felt_time(tmp_path, _DeferringProvider(), object(), 0.0)
    recs = [r for r in caplog.records if "heartbeat" in r.getMessage()]
    assert recs and all(r.levelno == logging.WARNING and r.exc_info is None for r in recs), [(r.levelno, r.getMessage()) for r in recs]


# C4b/C15 through the REAL composition: build_tier_provider → ClaudeCliProvider → gated generate
def test_expired_heartbeat_spawns_one_probe_and_still_persists(tmp_path: Path, monkeypatch, _no_real_tier_providers) -> None:
    from unittest.mock import MagicMock, patch

    from brain.bridge import model_tier, provider_auth

    # undo the autouse fake: use the real tier builder against a claude-cli persona
    monkeypatch.setattr("brain.engines.heartbeat.build_tier_provider", model_tier.build_tier_provider)
    (tmp_path / "persona_config.json").write_text(json.dumps({"provider": "claude-cli"}))
    t0 = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    clock = _Clock(t0)
    monkeypatch.setattr("brain.engines.heartbeat.datetime", _fake_datetime(clock))
    HeartbeatConfig(dream_every_hours=24, emit_memory="always").save(tmp_path / "hb_config.json")
    st = HeartbeatState.fresh("manual")
    st.last_tick_at = t0 - timedelta(minutes=15)
    st.last_dream_at = t0 - timedelta(hours=25)
    st.save(tmp_path / "heartbeat_state.json")

    failing = MagicMock()
    failing.returncode = 1
    failing.stdout = json.dumps({"is_error": True, "result": "Failed to authenticate: OAuth session expired and could not be refreshed"})
    failing.stderr = ""
    run = MagicMock(return_value=failing)

    from brain.bridge.provider import FakeProvider

    eng, store, hm = _engine(tmp_path, FakeProvider())
    try:
        with patch("subprocess.run", run):
            eng.run_tick(trigger="manual", dry_run=False)  # dream spawns once → expired; emit deferred
            clock.now_value = t0 + timedelta(minutes=15)
            eng.run_tick(trigger="manual", dry_run=False)  # dream + emit both deferred, no spawn
    finally:
        store.close()
        hm.close()
    assert run.call_count == 1, "exactly one probe spawn across two expired ticks"
    assert provider_auth.state()["status"] == "expired"
    assert HeartbeatState.load(tmp_path / "heartbeat_state.json").last_tick_at == t0 + timedelta(minutes=15)
