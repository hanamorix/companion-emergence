"""Integration test: supervisor sweeps expired pending file-writes on the
maintenance cadence (alongside forgetting + narrative).

Entry point under test: brain.bridge.supervisor.run_folded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def test_supervisor_sweeps_expired_pending_writes_on_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_folded calls pending.sweep_expired at least once on the maintenance tick."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    sweep_calls: list = []
    stop_event = threading.Event()

    def _fake_sweep(persona_dir, *, now):
        sweep_calls.append(persona_dir)
        stop_event.set()
        return 0

    monkeypatch.setattr("brain.files.pending.sweep_expired", _fake_sweep)
    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", lambda *a, **k: {})
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())
    # #154: voice-reflection (background-generative tier) now builds its own
    # real Sonnet-tier provider, and calls the LLM unconditionally before its
    # own evidence gate — a real-subprocess hazard on this bare tmp_path
    # persona (no persona_config.json); not about this test, neutralise it.
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_voice_reflection_tick", lambda *a, **k: None
    )

    # Watchdog: if the sweep never fires, stop the loop after a short window so
    # the test fails on the assertion below rather than hanging forever.
    threading.Timer(3.0, stop_event.set).start()

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=0.05,
        finalize_interval_s=None,
        # #154: interest_sweep now builds its own real Haiku-tier provider
        # (persona_dir here has no persona_config.json, so it would default to
        # a real ClaudeCliProvider and attempt a genuine subprocess call) —
        # disabled, out of scope for this pending-sweep-focused test. The
        # 3s watchdog above does not save this: a blocking subprocess call
        # inside one iteration isn't interrupted by stop_event until that
        # call returns.
        interest_sweep_interval_s=None,
    )

    assert len(sweep_calls) >= 1


# ---------------------------------------------------------------------------
# #344 — a write stranded in 'committing' is reconciled by the supervisor:
# once at startup (1-minute gate) and on every maintenance tick (10-minute gate).
# Criteria: docs/guarded-change/pending-committing-recovery/1.5-criteria.md
# ---------------------------------------------------------------------------


def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise every provider-spawning tick so run_folded is safe on a bare tmp persona."""
    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", lambda *a, **k: {})
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_voice_reflection_tick", lambda *a, **k: None
    )


def _run(persona_dir: Path, stop_event: threading.Event, **overrides) -> None:
    run_folded(
        stop_event,
        **{
            "persona_dir": persona_dir,
            "provider": MagicMock(),
            "event_bus": MagicMock(),
            "tick_interval_s": 0.05,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": 0.05,
            "finalize_interval_s": None,
            "interest_sweep_interval_s": None,
            **overrides,
        },
    )


def _stranded(persona_dir: Path, target: Path, *, age_s: float, op: str = "create") -> str:
    """A record claimed `age_s` ago whose write never happened (crash between claim and write)."""
    from datetime import UTC, datetime, timedelta

    from brain.files import pending

    claimed = datetime.now(UTC) - timedelta(seconds=age_s)
    rid = pending.create(persona_dir, op=op, resolved_path=str(target), content="body", now=claimed)
    assert pending.mark(persona_dir, rid, status="committing", claimed_at=claimed.isoformat())
    return rid


def test_supervisor_startup_reconciles_a_stranded_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a crash + quick restart the claim is minutes old, not 10: the startup call uses a
    1-minute gate so it is recovered at once. A 30-second-old claim could still be in flight."""
    from brain.files import pending

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    old = _stranded(persona_dir, out / "a.md", age_s=120)
    young = _stranded(persona_dir, out / "b.md", age_s=30)
    _quiet(monkeypatch)

    stop_event = threading.Event()
    stop_event.set()  # the startup one-shots run before the loop; exit straight after them
    _run(persona_dir, stop_event, soul_review_interval_s=None)

    assert pending.get(persona_dir, old)["status"] == "error"
    assert pending.get(persona_dir, young)["status"] == "committing"


def _spy_reconcile(monkeypatch, stop_event, stage):
    """Wrap the REAL reconcile. Call 1 is the startup one-shot: delegate, THEN `stage()` the
    stranded record, so the startup call cannot be what resolves it. The next call can only come
    from the maintenance job; after it delegates, stop the loop. Returns the list of stale_after
    values seen (None = the default 10-minute gate)."""
    from brain.files import commit as commit_mod

    real, seen = commit_mod.reconcile_stale_commits, []

    def spy(persona_dir, **kw):
        n = real(persona_dir, **kw)
        seen.append(kw.get("stale_after"))
        if len(seen) == 1:
            stage(persona_dir)
        else:
            stop_event.set()
        return n

    monkeypatch.setattr(commit_mod, "reconcile_stale_commits", spy)
    threading.Timer(8.0, stop_event.set).start()  # watchdog: fail on the assertion, don't hang
    return seen


@pytest.mark.parametrize("op", ["create", "append"])
def test_maintenance_tick_reconciles_a_landed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, op: str
) -> None:
    """The issue's scenario end to end: real commit_write lands the file, its final 'committed'
    mark fails, the process 'dies'. The maintenance tick (not startup) must resolve it."""
    import json
    from datetime import UTC, datetime, timedelta

    from brain.files import pending
    from brain.files.commit import commit_write
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    out = tmp_path / "home" / "out"
    out.mkdir(parents=True)
    target = out / "n.md"
    if op == "append":
        target.write_text("seed\n", encoding="utf-8")
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    staged: dict = {}

    def stage(pd):
        rid = pending.create(pd, op=op, resolved_path=str(target.resolve()), content="BLOCK",
                             now=datetime.now(UTC))
        real_mark = pending.mark
        with monkeypatch.context() as m:  # every 'committed' mark fails; reconcile's must work
            m.setattr(pending, "mark", lambda p, r, *, status, **kw:
                      False if status == "committed" else real_mark(p, r, status=status, **kw))
            store = MemoryStore(pd / "memories.db")
            try:
                assert commit_write(pd, rid, store=store)["ok"]
            finally:
                store.close()
        assert pending.get(pd, rid)["status"] == "committing"
        pending.mark(pd, rid, status="committing",
                     claimed_at=(datetime.now(UTC) - timedelta(minutes=11)).isoformat())
        staged["rid"] = rid

    _quiet(monkeypatch)
    stop_event = threading.Event()
    seen = _spy_reconcile(monkeypatch, stop_event, stage)
    _run(persona_dir, stop_event)

    assert len(seen) >= 2, "the maintenance job never called reconcile"
    assert seen[0] is not None and seen[1] is None  # startup gate, then the default 10-min gate
    rec = pending.get(persona_dir, staged["rid"])
    assert rec["status"] == "committed" and rec["resolved_by"] == "reconcile"
    events = [json.loads(line)["event"]
              for line in (persona_dir / "write_audit.jsonl").read_text("utf-8").splitlines()]
    assert events.count("commit_reconciled") == 1
    wired = [m for m in PendingQueue(persona_dir).read_recent("file_write", limit=50)
             if str(target.resolve()) in m.content]
    assert len(wired) == 1, "reconcile must wire the memory exactly once"


def test_maintenance_tick_abandons_a_write_that_never_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash between the claim and the write (the common case): the target does not exist.
    The record must leave 'committing' as an audited 'error' — and must never be retried."""
    import json
    from datetime import UTC, datetime

    from brain.files import pending
    from brain.files.commit import commit_write
    from brain.memory.store import MemoryStore

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    out = tmp_path / "home" / "out"
    out.mkdir(parents=True)
    target = out / "never.md"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    staged: dict = {}

    def stage(pd):
        staged["rid"] = _stranded(pd, target.resolve(), age_s=11 * 60)

    _quiet(monkeypatch)
    stop_event = threading.Event()
    seen = _spy_reconcile(monkeypatch, stop_event, stage)
    _run(persona_dir, stop_event)

    assert len(seen) >= 2, "the maintenance job never called reconcile"
    rid = staged["rid"]
    rec = pending.get(persona_dir, rid)
    assert rec["status"] == "error" and rec["resolved_by"] == "reconcile"
    rows = [json.loads(line)
            for line in (persona_dir / "write_audit.jsonl").read_text("utf-8").splitlines()]
    assert [r["event"] for r in rows if r["id"] == rid] == ["commit_abandoned"]
    assert not target.exists()
    store = MemoryStore(persona_dir / "memories.db")
    try:
        assert commit_write(persona_dir, rid, store=store) == {
            "ok": False, "error": "not a pending write"}
    finally:
        store.close()
    assert pending.list_pending(persona_dir, now=datetime.now(UTC)) == []
    assert not target.exists()


def test_a_raising_reconcile_never_stops_startup_or_the_maintenance_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-soft: the sidecar sweep AFTER reconcile in the maintenance job must still run."""
    from brain.files import commit as commit_mod

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _quiet(monkeypatch)
    stop_event = threading.Event()
    boom_calls: list = []

    def boom(*a, **k):
        boom_calls.append(1)
        raise RuntimeError("reconcile exploded")

    swept: list = []

    def sidecar_sweep(pd, *, now):
        swept.append(pd)
        stop_event.set()
        return 0

    monkeypatch.setattr(commit_mod, "reconcile_stale_commits", boom)
    monkeypatch.setattr("brain.health.sidecar_sweep.sweep_stale_sidecars", sidecar_sweep)
    threading.Timer(8.0, stop_event.set).start()
    _run(persona_dir, stop_event)

    assert len(boom_calls) >= 2, "reconcile should have been attempted at startup AND on maintenance"
    assert swept, "a raising reconcile stopped the maintenance job before the sidecar sweep"


def test_reconcile_has_its_own_persisted_cadence_independent_of_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#346: the maintenance path shares the throttle-slot / PAUSED / 6h gates. A short own
    cadence recovers a write stranded mid-session, with the maintenance job switched OFF."""
    from brain.files import commit as commit_mod
    from brain.files import pending

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    staged: dict = {}
    _quiet(monkeypatch)
    stop_event = threading.Event()
    real, seen = commit_mod.reconcile_stale_commits, []

    def spy(pd, **kw):
        n = real(pd, **kw)
        seen.append(kw.get("stale_after"))
        if len(seen) == 1:  # startup done: strand a record the startup call cannot have seen
            staged["rid"] = _stranded(pd, (out / "n.md").resolve(), age_s=11 * 60)
        elif n:
            stop_event.set()
        return n

    monkeypatch.setattr(commit_mod, "reconcile_stale_commits", spy)
    threading.Timer(8.0, stop_event.set).start()
    _run(persona_dir, stop_event, soul_review_interval_s=None, pending_reconcile_interval_s=0.05)

    assert pending.get(persona_dir, staged["rid"])["status"] == "error"
    assert (persona_dir / "cadence" / "pending_reconcile_cadence.json").exists()


def test_pending_reconcile_cadence_default_is_fifteen_minutes() -> None:
    """The default IS the recovery latency for a write stranded mid-session; don't let it drift."""
    import inspect

    default = inspect.signature(run_folded).parameters["pending_reconcile_interval_s"].default
    assert default == 15 * 60.0
