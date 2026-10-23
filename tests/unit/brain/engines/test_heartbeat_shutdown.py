"""A heartbeat tick during a stop makes no further model call after the stop (C4, D2).

The stop flag is set while the tick's first model-calling stage (the dream) has its call in flight. Every later stage
that would call the model (reflex, research, growth, the heartbeat memory) is refused before any `claude` process is
spawned; the tick still saves a parseable `heartbeat_state.json`, and the cadence of every stage that was skipped is not
advanced.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.bridge.provider import ClaudeCliProvider
from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from tests.unit.brain.engines.test_heartbeat import DEFAULT_REFLEX_ARCS_PATH


@pytest.fixture(autouse=True)
def _clean():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


def test_flag_set_during_the_first_model_stage_stops_every_later_model_call(tmp_path: Path, monkeypatch):
    spawned: list[str] = []

    def fake_run(cmd, *a, **kw):
        spawned.append(kw.get("input", "")[:40])
        cli_throttle.request_shutdown()  # the stop arrives while this (first) call is in flight
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "DREAM: the sea at night"}),
                                           stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = ClaudeCliProvider(probe_model="claude-haiku-4-5")
    monkeypatch.setattr("brain.engines.heartbeat.build_tier_provider", lambda *a, **k: provider)
    monkeypatch.setattr("brain.bridge.model_tier.build_tier_provider", lambda *a, **k: provider)

    arcs = tmp_path / "arcs.json"
    arcs.write_text(json.dumps({"version": 1, "arcs": [{
        "name": "test_arc", "description": "d", "trigger": {"love": 5}, "days_since_human_min": 0,
        "cooldown_hours": 1.0, "action": "a", "output_memory_type": "reflex_journal", "prompt_template": "Hi.",
    }]}), encoding="utf-8")
    interests = tmp_path / "interests.json"
    interests.write_text(json.dumps({"interests": []}), encoding="utf-8")
    (tmp_path / "persona_config.json").write_text("{}", encoding="utf-8")
    HeartbeatConfig(dream_every_hours=1, emit_memory="always", growth_every_hours=1).save(tmp_path / "hb_config.json")
    t0 = datetime.now(UTC)
    st = HeartbeatState.fresh("manual")
    st.last_tick_at = t0 - timedelta(hours=2)
    st.last_dream_at = t0 - timedelta(hours=2)
    st.last_growth_at = t0 - timedelta(hours=2)
    st.save(tmp_path / "heartbeat_state.json")

    store = MemoryStore(tmp_path / "memories.db")
    hm = HebbianMatrix(":memory:")
    store.create(Memory.create_new(content="a seed", memory_type="conversation", domain="us", emotions={"love": 8.0}))
    eng = HeartbeatEngine(
        store=store, hebbian=hm, provider=provider,
        state_path=tmp_path / "heartbeat_state.json", config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl", heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        persona_name="Nell", persona_system_prompt="You are Nell.",
        reflex_arcs_path=arcs, reflex_log_path=tmp_path / "reflex_log.json",
        reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
        interests_path=interests, research_log_path=tmp_path / "research.log.jsonl",
    )
    try:
        eng.run_tick(trigger="manual", dry_run=False)
    finally:
        store.close()
        hm.close()

    assert len(spawned) == 1, f"a model call was spawned after the stop: {spawned}"
    after = HeartbeatState.load(tmp_path / "heartbeat_state.json")
    assert after is not None  # parseable
    assert after.last_growth_at == st.last_growth_at  # growth skipped: its cadence is not advanced
