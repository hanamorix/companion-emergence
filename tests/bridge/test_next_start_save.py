"""The next start saves what the last bridge left (C36, C37; D12, Q-B "c", P1 "That's fine").

No shutdown step saves sessions any more. The existing gated job `session_snapshot_prune` does it at the next start's
first central pass with chat idle, inside the background slot; `supervisor start` / `run` run no synchronous recovery.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.bridge import central_cadence, cli_throttle, daemon, state_file
from brain.bridge.events import EventBus
from brain.bridge.provider import LLMProvider

PREVIOUS = ["normal_stop", "hard_exit", "sigkill", "471"]


class _Extractor(LLMProvider):
    """Extracts one memory per user turn it is shown, and records every turn text it saw."""

    def __init__(self):
        self.seen: list[str] = []

    def generate(self, prompt: str, *, system=None, **_kw) -> str:
        items = []
        for line in prompt.splitlines():
            if "turn-" in line:
                tag = line[line.index("turn-"):].split()[0].strip(".,:;\"'")
                self.seen.append(tag)
                items.append({"text": f"remembered {tag}", "label": "fact", "importance": 3, "emotions": {}})
        return json.dumps(items)

    def name(self) -> str:
        return "extractor"

    def chat(self, messages, *, tools=None, options=None):  # pragma: no cover
        raise NotImplementedError


def _record(pd: Path, previous: str) -> None:
    base = {"persona": "nell", "port": None, "started_at": "t", "client_origin": "cli"}
    shapes = {
        "normal_stop": {"pid": None, "stopped_at": "t2", "shutdown_clean": True, "phase": "stopped"},
        "hard_exit": {"pid": None, "stopped_at": None, "shutdown_clean": False, "phase": "hard_exit",
                      "stop_reason": "hard_exit"},
        "sigkill": {"pid": 999_999, "stopped_at": None, "shutdown_clean": False, "phase": "running",
                    "pid_token": "linux:x:999999:1", "identity": "token"},
        "471": {"pid": None, "stopped_at": "t2", "shutdown_clean": False, "drain_errors": 2},
    }
    state_file.write(pd, state_file.BridgeState(**base, **shapes[previous]))


def _sessions(pd: Path, *, age: timedelta) -> list[str]:
    from brain.ingest.buffer import ingest_turn

    tags = []
    t0 = datetime.now(UTC) - age
    for s in range(2):
        for i in range(3):
            tag = f"turn-{s}-{i}"
            tags.append(tag)
            ingest_turn(pd, {"session_id": f"sess_{s}", "speaker": "user", "text": f"{tag} about the sea",
                             "ts": (t0 + timedelta(seconds=s * 10 + i)).isoformat()})
    return tags


@pytest.fixture
def persona(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    pd = tmp_path / "home" / "personas" / "nell"
    (pd / "active_conversations").mkdir(parents=True)
    (pd / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}', encoding="utf-8")
    cli_throttle.reset()
    yield pd
    cli_throttle.reset()


def _first_pass(pd: Path, provider: LLMProvider, monkeypatch) -> list:
    import brain.bridge.supervisor as sup

    monkeypatch.setattr(sup, "build_tier_provider", lambda *_a, **_k: provider)
    cli_throttle.seed_last_message_from_active_conversations(pd)
    jobs = sup._build_gated_jobs(
        persona_dir=pd, provider=provider, event_bus=EventBus(), is_session_busy=None,
        finalize_after_hours=24.0, finalize_interval_s=None, initiate_review_interval_s=None,
        maintenance_interval_s=None, self_model_interval_s=None, compaction_interval_s=None,
        calibration_interval_s=None, interest_sweep_interval_s=None, judge_selftune_interval_s=None,
        clustering_interval_s=None, intensity_drivers=lambda: None,
        tick_stats={"closed_sessions": 0, "pruned_empty_sessions": 0},
    )
    jobs = [j for j in jobs if j.name == "session_snapshot_prune"]
    return central_cadence.run_central_pass(pd, jobs)


@pytest.mark.parametrize("previous", PREVIOUS)
def test_the_first_idle_pass_saves_every_open_session(persona, monkeypatch, previous):
    """C36: whatever ended the last bridge, the first central pass with chat idle extracts every open session's turns
    past its cursor, through the gated job, inside the background slot."""
    from brain.ingest.buffer import read_cursor, read_session

    _record(persona, previous)
    tags = _sessions(persona, age=timedelta(hours=2))
    provider = _Extractor()
    slots: list[bool] = []
    real_slot = cli_throttle.background_slot

    from contextlib import contextmanager

    @contextmanager
    def _spy_slot(*a, **k):
        with real_slot(*a, **k) as got:
            slots.append(got)
            yield got

    monkeypatch.setattr(cli_throttle, "background_slot", _spy_slot)
    decisions = _first_pass(persona, provider, monkeypatch)
    assert ("session_snapshot_prune", "completed") in {(d.job, d.action) for d in decisions}
    assert sorted(set(provider.seen)) == sorted(tags)
    for sid in ("sess_0", "sess_1"):
        assert read_cursor(persona, sid) == read_session(persona, sid)[-1]["ts"]


def test_the_job_waits_for_the_lull(persona, monkeypatch):
    """C36: with the last message younger than the lull, the job does not run yet."""
    _record(persona, "sigkill")
    _sessions(persona, age=timedelta(seconds=5))
    provider = _Extractor()
    decisions = _first_pass(persona, provider, monkeypatch)
    assert provider.seen == []
    assert ("session_snapshot_prune", "run") not in {(d.job, d.action) for d in decisions}


@pytest.mark.parametrize("command", ["start", "run"])
@pytest.mark.parametrize("previous", PREVIOUS)
def test_supervisor_start_and_run_run_no_synchronous_recovery(persona, monkeypatch, command, previous):
    """C36: no recovery inside `supervisor start` / `run` (spy on the snapshot)."""
    from brain.ingest import pipeline

    _record(persona, previous)
    _sessions(persona, age=timedelta(hours=2))
    called: list[str] = []
    monkeypatch.setattr(pipeline, "snapshot_stale_sessions", lambda *a, **k: called.append("snapshot") or [])
    if hasattr(daemon, "snapshot_stale_sessions"):
        monkeypatch.setattr(daemon, "snapshot_stale_sessions", lambda *a, **k: called.append("snapshot") or [])
    args = SimpleNamespace(persona="nell", idle_shutdown=0, client_origin="tests")
    if command == "start":
        monkeypatch.setattr(daemon, "spawn_detached", lambda *a, **k: 4242)
        monkeypatch.setattr(daemon, "_await_readiness", lambda *a, **k: 0)
        assert daemon.cmd_start(args) == 0
    else:
        monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", lambda pd, **kw: 0)
        assert daemon.cmd_run(args) == 0
    assert called == []


class _Killed(BaseException):
    """Stands for the process ending mid-job (a stop, a hard exit or a SIGKILL): nothing after it runs."""


def test_nothing_is_lost_over_repeated_interrupted_restarts(persona, monkeypatch):
    """C37 (gating half): stop / hard exit / SIGKILL during the snapshot job, three times each, then a complete start:
    every turn of every session is extracted at least once. (Advisory half: no memory stored twice.)"""
    from brain.memory.store import MemoryStore

    tags = _sessions(persona, age=timedelta(hours=2))
    seen_all: list[str] = []

    class _DiesAfterFirst(_Extractor):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, prompt, *, system=None, **kw):
            self.calls += 1
            if self.calls > 1:
                raise _Killed
            out = super().generate(prompt, system=system)
            seen_all.extend(self.seen)
            return out

    from brain.ingest.buffer import ingest_turn

    killed = 0
    for n, _ending in enumerate(("stop", "hard_exit", "sigkill") * 3):
        # the user chatted again before this restart: both sessions have new turns past their cursors
        for s in range(2):
            tag = f"turn-{s}-r{n}"
            tags.append(tag)
            ingest_turn(persona, {"session_id": f"sess_{s}", "speaker": "user", "text": f"{tag} about the sea",
                                  "ts": (datetime.now(UTC) - timedelta(hours=1, seconds=-n)).isoformat()})
        cli_throttle.reset()
        try:
            _first_pass(persona, _DiesAfterFirst(), monkeypatch)
        except _Killed:
            killed += 1  # the process ended part-way through the job
    assert killed == 9  # every restart ended part-way through the job
    cli_throttle.reset()
    final = _Extractor()
    _first_pass(persona, final, monkeypatch)
    assert set(seen_all) | set(final.seen) == set(tags)
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        contents = [m.content for m in store.list_active(limit=None)]
    finally:
        store.close()
    assert len(contents) == len(set(contents))  # advisory: no memory stored twice
