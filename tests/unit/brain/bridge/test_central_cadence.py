"""ram-spike-fix INC-9 — the central cadence function (spec §3; S16/S20/S22/
S23/S29/S34/S41/S43/S44/S53/S55/S65/S66/S69/S70/S72/S73/S83).

Criteria covered here: C5, C6, C7, C15(a)(b), C21, C22, C23 (INC-9 parts),
C26, C38, C39. Deterministic by construction: injected wall clock
(``now_wall``), injected idle/slot answers or a fake monotonic ``now`` fed to
the real ``cli_throttle.is_chat_idle``, and job bodies replaced by recorders —
no sleeps with tight margins, no real LLM/model, synthetic tmp personas only.

"Real table" tests build the production job table with
``supervisor._build_gated_jobs`` and stub only the leaf tick functions each
job calls, so the job wiring (names, order, due sources, cadence files,
outcomes) is the code under test, not a proxy.
"""
from __future__ import annotations

import ast
import inspect
import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.bridge import central_cadence, cli_throttle, persisted_cadence, supervisor
from brain.bridge.central_cadence import GATED_JOB_ORDER, GatedJob, JobOutcome
from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider
from brain.chat import pass2_queue
from brain.engines import interest_sweep
from brain.memory.judge_selftune import JUDGE_TUNE_CADENCE_FILE
from brain.self_model import cadence as sm_cadence

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)

# Interval jobs → (cadence file, the interval _real_jobs() gives them).
INTERVAL_JOBS: dict[str, tuple[str, float]] = {
    "maintenance": ("maintenance_cadence.json", 6 * 3600.0),
    "interest_sweep": (interest_sweep.SWEEP_CADENCE_FILE, 7 * 86400.0),
    "compaction": ("compaction_cadence.json", 86400.0),
    "clustering": ("clustering_cadence.json", 6 * 3600.0),
    "daily_calibration": ("calibration_cadence.json", 86400.0),
    "weekly_selftune": (JUDGE_TUNE_CADENCE_FILE, 7 * 86400.0),
    "finalize": ("finalize_cadence.json", 3600.0),
    "initiate_review": ("initiate_review_cadence.json", 900.0),
}
NO_INTERVAL_JOBS = ("pass2", "session_snapshot_prune", "emotion_backfill", "embedding_backfill")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')
    return p


def _cadence_path(persona_dir: Path, filename: str) -> Path:
    return persona_dir / "cadence" / filename


def _seed(persona_dir: Path, filename: str, next_at: datetime) -> None:
    persisted_cadence.save_cadence(
        persona_dir, filename, persisted_cadence.CadenceState(next_at=next_at)
    )


def _seed_all_overdue(persona_dir: Path) -> None:
    for filename, _interval in INTERVAL_JOBS.values():
        _seed(persona_dir, filename, NOW - timedelta(hours=1))
    sm_cadence.save(
        persona_dir,
        sm_cadence.SelfModelCadenceState(
            next_reflection_at=datetime.now(UTC) - timedelta(hours=1), consecutive_failures=0
        ),
    )


def _snapshot_bytes(persona_dir: Path) -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {}
    for filename, _i in INTERVAL_JOBS.values():
        path = _cadence_path(persona_dir, filename)
        out[filename] = path.read_bytes() if path.exists() else None
    sm = sm_cadence._state_path(persona_dir)  # noqa: SLF001
    out["self_model"] = sm.read_bytes() if sm.exists() else None
    return out


def _real_jobs(persona_dir: Path, **overrides) -> list[GatedJob]:
    kwargs = {
        "persona_dir": persona_dir,
        "provider": FakeProvider(),
        "event_bus": EventBus(),
        "is_session_busy": None,
        "finalize_after_hours": 24.0,
        "finalize_interval_s": INTERVAL_JOBS["finalize"][1],
        "initiate_review_interval_s": INTERVAL_JOBS["initiate_review"][1],
        "maintenance_interval_s": INTERVAL_JOBS["maintenance"][1],
        "self_model_interval_s": 0.0,
        "compaction_interval_s": INTERVAL_JOBS["compaction"][1],
        "calibration_interval_s": INTERVAL_JOBS["daily_calibration"][1],
        "interest_sweep_interval_s": INTERVAL_JOBS["interest_sweep"][1],
        "judge_selftune_interval_s": INTERVAL_JOBS["weekly_selftune"][1],
        "clustering_interval_s": INTERVAL_JOBS["clustering"][1],
        "intensity_drivers": lambda: None,
        "tick_stats": {"closed_sessions": 0, "pruned_empty_sessions": 0},
    }
    kwargs.update(overrides)
    return supervisor._build_gated_jobs(**kwargs)


@contextmanager
def _granting_slot(*, now=None):  # noqa: ARG001
    yield True


@contextmanager
def _denying_slot(*, now=None):  # noqa: ARG001
    yield False


@pytest.fixture
def stubs(monkeypatch):
    """Replace every real job's LEAF work with a recorder (order list) and make
    every no-interval/predicate job report work. The job table itself, the
    central function and the cadence handling stay real."""
    order: list[str] = []
    work = dict.fromkeys((*NO_INTERVAL_JOBS, "deploy_recalibration"), True)

    def rec(name, ret=None):
        def _f(*_a, **_k):
            order.append(name)
            return ret

        return _f

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    monkeypatch.setattr(supervisor, "build_tier_provider", lambda *a, **k: FakeProvider())
    monkeypatch.setattr(pass2_queue, "queue_length", lambda _pd: 1 if work["pass2"] else 0)

    def _pass2_drain(*_a, **_k):
        # INC-10: _pass2_run now checks queue_length() AFTER draining to tell
        # a real pause (items remain) from a full drain — simulate the queue
        # actually emptying (one item, fully drained) so this stub's job
        # reports COMPLETED like every other stubbed job here, not PAUSED.
        order.append("pass2")
        work["pass2"] = False
        return 1

    monkeypatch.setattr(pass2_queue, "drain_all_locked", _pass2_drain)
    monkeypatch.setattr(supervisor, "_snapshot_has_work", lambda _pd: work["session_snapshot_prune"])
    monkeypatch.setattr(supervisor, "snapshot_stale_sessions", rec("session_snapshot_prune", []))
    monkeypatch.setattr(
        supervisor, "_emotion_backfill_has_work", lambda _pd, **_k: work["emotion_backfill"]
    )
    # INC-10: _emotion_backfill_job reads .status off the return value to
    # tell a real between-items pause ("running") from a finish.
    monkeypatch.setattr(
        supervisor, "_emotion_backfill_run", rec("emotion_backfill", SimpleNamespace(status="complete"))
    )
    monkeypatch.setattr(
        supervisor, "_embedding_backfill_has_work_probe", lambda _s: work["embedding_backfill"]
    )
    monkeypatch.setattr(
        supervisor,
        "_embedding_backfill_run_tick",
        rec(
            "embedding_backfill",
            SimpleNamespace(scanned=0, embedded=0, skipped_short=0, errors=0, batch_size=1),
        ),
    )
    monkeypatch.setattr(supervisor, "forgetting_run_pass", rec("maintenance"))
    monkeypatch.setattr(supervisor, "_run_narrative_memory_pass", lambda *a, **k: None)
    monkeypatch.setattr("brain.files.pending.sweep_expired", lambda *a, **k: None)
    monkeypatch.setattr("brain.health.sidecar_sweep.sweep_stale_sidecars", lambda *a, **k: None)
    monkeypatch.setattr(interest_sweep, "run_sweep_tick", rec("interest_sweep"))
    monkeypatch.setattr(
        "brain.self_model.articulate.build_self_model_provider", lambda _pd: FakeProvider()
    )
    monkeypatch.setattr(supervisor, "_run_self_model_tick", rec("self_model_articulation", True))
    monkeypatch.setattr(
        "brain.chat.compaction.build_compaction_provider", lambda _pd: FakeProvider()
    )
    monkeypatch.setattr(supervisor, "_run_compaction_tick", rec("compaction"))
    monkeypatch.setattr(supervisor, "_run_clustering_tick", rec("clustering"))
    monkeypatch.setattr(
        supervisor, "_deploy_recalibration_due", lambda _pd: work["deploy_recalibration"]
    )
    monkeypatch.setattr(supervisor, "_run_deploy_recalibration", rec("deploy_recalibration"))
    monkeypatch.setattr(supervisor, "_run_calibration_tick", rec("daily_calibration", True))
    monkeypatch.setattr(supervisor, "_run_judge_selftune_tick", rec("weekly_selftune"))
    monkeypatch.setattr(supervisor, "_run_finalize_tick", rec("finalize"))
    monkeypatch.setattr(supervisor, "_run_initiate_review_tick", rec("initiate_review"))
    return SimpleNamespace(order=order, work=work)


def _pass(persona_dir, jobs, *, idle=True, slot=True, now=NOW, between_jobs=None):
    return central_cadence.run_central_pass(
        persona_dir,
        jobs,
        is_idle=(idle if callable(idle) else (lambda: idle)),
        slot_available=(slot if callable(slot) else (lambda: slot)),
        now_wall=lambda: now,
        between_jobs=between_jobs,
    )


def _fake_jobs(names, log, *, due=True):
    return [
        GatedJob(n, run=(lambda n=n: log.append(n)), has_work=(lambda: due)) for n in names
    ]


# ---------------------------------------------------------------------------
# The job table (S16/S55/S70/S73)
# ---------------------------------------------------------------------------


def test_job_table_is_the_14_gated_jobs_in_the_s55_order(tmp_path):
    names = [j.name for j in _real_jobs(_persona(tmp_path))]
    assert names == list(GATED_JOB_ORDER)
    assert len(names) == 14
    # C38: deploy recalibration immediately after clustering, immediately
    # before daily calibration; C15/S43: calibration before self-tune.
    i = names.index("deploy_recalibration")
    assert names[i - 1] == "clustering" and names[i + 1] == "daily_calibration"
    assert names.index("daily_calibration") < names.index("weekly_selftune")
    # The heartbeat is not a gated job (S16/S21).
    assert "heartbeat" not in names


def test_job_table_due_sources(tmp_path):
    jobs = {j.name: j for j in _real_jobs(_persona(tmp_path))}
    for name, (filename, interval) in INTERVAL_JOBS.items():
        assert jobs[name].cadence_file == filename and jobs[name].interval_s == interval
    # No-interval jobs (S53/S66) and the predicate jobs (S70 deploy, S29
    # self-model) own no cadence file here.
    for name in (*NO_INTERVAL_JOBS, "deploy_recalibration", "self_model_articulation"):
        assert jobs[name].cadence_file is None and jobs[name].has_work is not None


def test_unknown_job_name_is_rejected():
    with pytest.raises(ValueError):
        central_cadence.order_jobs([GatedJob("dream", run=lambda: None, has_work=lambda: True)])


# ---------------------------------------------------------------------------
# C15(a) — S55 order, calibration before self-tune, re-check before each
# ---------------------------------------------------------------------------


def test_c15a_all_14_due_run_in_s55_order_real_table(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    decisions = _pass(persona_dir, _real_jobs(persona_dir))
    assert stubs.order == list(GATED_JOB_ORDER)
    runs = [d.job for d in decisions if d.action == "run"]
    assert runs == list(GATED_JOB_ORDER)


@pytest.mark.parametrize("j", range(len(GATED_JOB_ORDER)))
def test_c15a_idle_flip_before_job_j_stops_the_sequence_at_j(tmp_path, j):
    log: list[str] = []
    asked: list[int] = []

    def is_idle():
        asked.append(1)
        return len(asked) <= j  # idle for the first j asks, then not

    decisions = _pass(tmp_path, _fake_jobs(GATED_JOB_ORDER, log), idle=is_idle)
    assert log == list(GATED_JOB_ORDER[:j])
    assert len(asked) == j + 1, "is-chat-idle must be asked again before every job"
    assert (decisions[-1].job, decisions[-1].action) == (GATED_JOB_ORDER[j], "skip-no-lull")


def test_c15a_idle_flip_real_table_stops_before_calibration(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    stop_at = GATED_JOB_ORDER.index("daily_calibration")
    asked: list[int] = []

    def is_idle():
        asked.append(1)
        return len(asked) <= stop_at

    before = _snapshot_bytes(persona_dir)
    _pass(persona_dir, _real_jobs(persona_dir), idle=is_idle)
    assert stubs.order == list(GATED_JOB_ORDER[:stop_at])
    after = _snapshot_bytes(persona_dir)
    # Jobs not reached keep their cadence (still due at the next lull).
    for name in ("daily_calibration", "weekly_selftune", "finalize", "initiate_review"):
        filename = INTERVAL_JOBS[name][0]
        assert after[filename] == before[filename], name


# ---------------------------------------------------------------------------
# C5 — the lull window (fake monotonic clock, real is_chat_idle)
# ---------------------------------------------------------------------------


def test_c5_no_gated_job_starts_inside_the_lull_then_all_start_after(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    jobs = _real_jobs(persona_dir)
    lull = cli_throttle._lull_seconds()  # noqa: SLF001
    t0 = 1_000_000.0
    cli_throttle.note_user_message(at=t0)
    cli_throttle.note_reply_end(at=t0)

    for t in (t0, t0 + lull / 2, t0 + lull - 0.001):
        _pass(persona_dir, jobs, idle=lambda t=t: cli_throttle.is_chat_idle(now=t))
        assert stubs.order == [], f"a gated job started inside the lull (t-t0={t - t0})"

    _pass(persona_dir, jobs, idle=lambda: cli_throttle.is_chat_idle(now=t0 + lull))
    assert stubs.order == list(GATED_JOB_ORDER), "all 14 due jobs start at the first pass after"


def test_c5_reply_in_flight_blocks_even_past_the_lull(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    cli_throttle.note_user_message(at=0.0)  # never ended: a reply is in flight
    _pass(persona_dir, _real_jobs(persona_dir), idle=lambda: cli_throttle.is_chat_idle(now=1e9))
    assert stubs.order == []
    cli_throttle.note_reply_end(at=0.0)


def test_c5_fresh_instance_runs_due_jobs_on_its_first_pass(tmp_path, stubs):
    """Bridge start, no message yet (anchor -inf, S23/S35): the very first
    pass of a fresh function instance is a lull, so overdue jobs run."""
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    cli_throttle.reset()
    _pass(persona_dir, _real_jobs(persona_dir), idle=lambda: cli_throttle.is_chat_idle(now=0.0))
    assert stubs.order == list(GATED_JOB_ORDER)


def test_c5_not_due_jobs_do_not_run_at_startup(tmp_path, stubs):
    """Fresh instance, idle, but nothing due: future cadences, no work, and a
    fresh install (no cadence files at all) — nothing runs on the first pass."""
    persona_dir = _persona(tmp_path)
    for name in stubs.work:
        stubs.work[name] = False
    for filename, _i in list(INTERVAL_JOBS.values())[:4]:
        _seed(persona_dir, filename, NOW + timedelta(hours=1))  # present, not due
    # the other 4 interval files and the self-model file are missing
    cli_throttle.reset()
    _pass(persona_dir, _real_jobs(persona_dir))
    assert stubs.order == []


def test_c5_run_folded_runs_an_overdue_job_at_the_bridge_start_lull(tmp_path, monkeypatch):
    """Live path: a real run_folded with an OVERDUE compaction cadence runs
    compaction on its first loop pass; with the cadence missing it does not
    (S22), and nothing runs before the loop (no startup catch-up, C26)."""
    import threading

    calls: list[str] = []
    stop = threading.Event()

    def fake_compaction(*_a, **_k):
        calls.append("compaction")

    monkeypatch.setattr(supervisor, "_run_compaction_tick", fake_compaction)
    monkeypatch.setattr(
        "brain.chat.compaction.build_compaction_provider", lambda _pd: FakeProvider()
    )
    ticks: list[int] = []

    def counting_heartbeatless_publish(event):
        if event.get("type") == "supervisor_tick":
            ticks.append(1)
            if len(ticks) >= 2:
                stop.set()

    bus = SimpleNamespace(publish=counting_heartbeatless_publish)
    common = {
        "provider": FakeProvider(),
        "event_bus": bus,
        "tick_interval_s": 0.0,
        "heartbeat_interval_s": None,
        "soul_review_interval_s": None,
        "finalize_interval_s": None,
        "log_rotation_interval_s": None,
        "initiate_review_interval_s": None,
        "voice_reflection_interval_s": None,
        "self_model_interval_s": None,
        "calibration_interval_s": None,
        "interest_sweep_interval_s": None,
        "judge_selftune_interval_s": None,
        "clustering_interval_s": None,
        "vocab_repair_interval_s": None,
        "maker_enabled": False,
        "notes_enabled": False,
        "kindled_link_enabled": False,
    }

    fresh = _persona(tmp_path)
    supervisor.run_folded(stop, persona_dir=fresh, **common)
    assert calls == [], "a missing cadence file means 'last ran now' — not due"
    assert _cadence_path(fresh, "compaction_cadence.json").exists()

    (tmp_path / "b").mkdir()
    overdue = _persona(tmp_path / "b")
    _seed(overdue, "compaction_cadence.json", datetime.now(UTC) - timedelta(days=2))
    stop.clear()
    ticks.clear()
    supervisor.run_folded(stop, persona_dir=overdue, **common)
    assert calls == ["compaction"], "overdue job runs once, at the first (bridge-start) lull"


# ---------------------------------------------------------------------------
# C6 — no lull / slot denied: cadence byte-identical, still due next pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deny", ["no-lull", "slot"])
def test_c6_skip_leaves_every_cadence_file_byte_identical_and_still_due(tmp_path, stubs, deny):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    jobs = _real_jobs(persona_dir)
    before = _snapshot_bytes(persona_dir)

    decisions = _pass(
        persona_dir, jobs, idle=(deny != "no-lull"), slot=(deny != "slot")
    )
    assert stubs.order == []
    assert _snapshot_bytes(persona_dir) == before
    if deny == "slot":
        assert [d.action for d in decisions if d.action != "init-cadence"] == ["skip-slot"] * 14

    _pass(persona_dir, jobs)  # next pass, idle + slot: every job still due
    assert stubs.order == list(GATED_JOB_ORDER)


def test_c6_inner_slot_denial_is_a_skip_not_a_run(tmp_path, stubs, monkeypatch):
    """Jobs holding their own authoritative slot (pass 2, embedding backfill,
    maintenance, interest sweep, self-tune) that lose the race after the
    central peek report SKIPPED: no work, no cadence advance."""
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _denying_slot)
    before = _snapshot_bytes(persona_dir)
    decisions = _pass(persona_dir, _real_jobs(persona_dir))
    skipped = {d.job for d in decisions if d.action == "skipped"}
    assert skipped == {
        "pass2", "embedding_backfill", "maintenance", "interest_sweep", "weekly_selftune"
    }
    after = _snapshot_bytes(persona_dir)
    for name in ("maintenance", "interest_sweep", "weekly_selftune"):
        filename = INTERVAL_JOBS[name][0]
        assert after[filename] == before[filename], name


def test_c6_calibration_deferred_for_a_busy_session_does_not_advance(tmp_path, stubs, monkeypatch):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    monkeypatch.setattr(supervisor, "_run_calibration_tick", lambda *a, **k: False)
    filename = INTERVAL_JOBS["daily_calibration"][0]
    before = _cadence_path(persona_dir, filename).read_bytes()
    decisions = _pass(persona_dir, _real_jobs(persona_dir))
    assert ("daily_calibration", "skipped") in [(d.job, d.action) for d in decisions]
    assert _cadence_path(persona_dir, filename).read_bytes() == before


def test_c6_completed_interval_job_advances_by_its_interval(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    _pass(persona_dir, _real_jobs(persona_dir))
    for name, (filename, interval) in INTERVAL_JOBS.items():
        state = persisted_cadence.load_cadence(persona_dir, filename)
        assert state.next_at == NOW + timedelta(seconds=interval), name


# ---------------------------------------------------------------------------
# C7 — missing / corrupt cadence file → "last ran now", not run; past → runs
# ---------------------------------------------------------------------------


_CORRUPT_VARIANTS = {
    "absent": None,
    "not-json": "{not json",
    "not-an-object": "[]",
    "missing-next_at": "{}",
    "null-next_at": '{"next_at": null}',
    "garbage-next_at": '{"next_at": "not a time"}',
}


@pytest.mark.parametrize("variant", list(_CORRUPT_VARIANTS))
def test_c7_missing_or_corrupt_cadence_is_created_as_last_ran_now(tmp_path, stubs, variant):
    persona_dir = _persona(tmp_path)
    for name in stubs.work:
        stubs.work[name] = False
    content = _CORRUPT_VARIANTS[variant]
    if content is not None:
        (persona_dir / "cadence").mkdir(exist_ok=True)
        for filename, _i in INTERVAL_JOBS.values():
            _cadence_path(persona_dir, filename).write_text(content)
        sm_cadence._state_path(persona_dir).write_text(content)  # noqa: SLF001

    wall_before = datetime.now(UTC)
    decisions = _pass(persona_dir, _real_jobs(persona_dir))
    assert stubs.order == [], "a missing/corrupt cadence must not run the job"
    for name, (filename, interval) in INTERVAL_JOBS.items():
        state = persisted_cadence.load_cadence(persona_dir, filename)
        assert state.next_at == NOW + timedelta(seconds=interval), name
        assert (name, "init-cadence") in [(d.job, d.action) for d in decisions]
    sm_state = sm_cadence.load(persona_dir)
    assert sm_state.next_reflection_at is not None
    assert sm_state.next_reflection_at >= wall_before + timedelta(hours=6) - timedelta(seconds=5)

    # The next pass (same clock) still does not run them: one full interval.
    _pass(persona_dir, _real_jobs(persona_dir))
    assert stubs.order == []
    # ...and one interval later they run.
    _pass(persona_dir, _real_jobs(persona_dir), now=NOW + timedelta(days=8))
    assert set(INTERVAL_JOBS) <= set(stubs.order)


def test_c7_present_past_cadence_runs_at_the_next_lull(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    for name in stubs.work:
        stubs.work[name] = False
    for filename, _i in INTERVAL_JOBS.values():
        _seed(persona_dir, filename, NOW - timedelta(days=30))
    _pass(persona_dir, _real_jobs(persona_dir), idle=False)
    assert stubs.order == []
    _pass(persona_dir, _real_jobs(persona_dir))
    assert stubs.order == [n for n in GATED_JOB_ORDER if n in INTERVAL_JOBS]


# ---------------------------------------------------------------------------
# C21 — a job that raises still advances its cadence (S44)
# ---------------------------------------------------------------------------


_RAISING_TARGETS = {
    "maintenance": "forgetting_run_pass",  # caught by the job's OWN handler
    "interest_sweep": None,  # patched on the interest_sweep module
    "compaction": "_run_compaction_tick",
    "clustering": "_run_clustering_tick",
    "daily_calibration": "_run_calibration_tick",
    "weekly_selftune": "_run_judge_selftune_tick",
    "finalize": "_run_finalize_tick",
    "initiate_review": "_run_initiate_review_tick",
}


@pytest.mark.parametrize("name", list(_RAISING_TARGETS))
def test_c21_a_raising_job_still_advances_its_cadence(tmp_path, stubs, monkeypatch, name):
    persona_dir = _persona(tmp_path)
    for k in stubs.work:
        stubs.work[k] = False
    filename, interval = INTERVAL_JOBS[name]
    _seed(persona_dir, filename, NOW - timedelta(hours=1))

    def boom(*_a, **_k):
        raise RuntimeError(f"{name} exploded")

    target = _RAISING_TARGETS[name]
    if target is None:
        monkeypatch.setattr(interest_sweep, "run_sweep_tick", boom)
    else:
        monkeypatch.setattr(supervisor, target, boom)
    decisions = _pass(persona_dir, _real_jobs(persona_dir))
    assert (name, "completed") in [(d.job, d.action) for d in decisions]
    state = persisted_cadence.load_cadence(persona_dir, filename)
    assert state.next_at == NOW + timedelta(seconds=interval)


def test_c21_self_model_reflect_raising_still_advances_its_own_cadence(tmp_path, monkeypatch):
    """Self-model keeps its own cadence (S29): a reflect that raises is
    caught by its own handler and backs off (still an advance, S44)."""
    persona_dir = _persona(tmp_path)
    sm_cadence.save(
        persona_dir,
        sm_cadence.SelfModelCadenceState(
            next_reflection_at=datetime.now(UTC) - timedelta(hours=1), consecutive_failures=0
        ),
    )
    monkeypatch.setattr(
        supervisor, "_self_model_reflect", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        "brain.self_model.articulate.build_self_model_provider", lambda _pd: FakeProvider()
    )
    jobs = [j for j in _real_jobs(persona_dir) if j.name == "self_model_articulation"]
    cli_throttle.reset()
    decisions = _pass(persona_dir, jobs)
    assert ("self_model_articulation", "completed") in [(d.job, d.action) for d in decisions]
    state = sm_cadence.load(persona_dir)
    assert state.consecutive_failures == 1
    assert state.next_reflection_at > datetime.now(UTC)


# ---------------------------------------------------------------------------
# C22 — self-model articulation: own cadence AND idle; skip leaves it unchanged
# ---------------------------------------------------------------------------


def _self_model_only(persona_dir):
    return [j for j in _real_jobs(persona_dir) if j.name == "self_model_articulation"]


def test_c22_not_due_by_its_own_cadence_does_not_run(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    sm_cadence.save(
        persona_dir,
        sm_cadence.SelfModelCadenceState(
            next_reflection_at=datetime.now(UTC) + timedelta(minutes=30), consecutive_failures=0
        ),
    )
    _pass(persona_dir, _self_model_only(persona_dir))
    assert stubs.order == []


def test_c22_due_but_no_lull_leaves_the_cadence_file_unchanged(tmp_path, stubs):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    before = sm_cadence._state_path(persona_dir).read_bytes()  # noqa: SLF001
    _pass(persona_dir, _self_model_only(persona_dir), idle=False)
    _pass(persona_dir, _self_model_only(persona_dir), slot=False)
    assert stubs.order == []
    assert sm_cadence._state_path(persona_dir).read_bytes() == before  # noqa: SLF001
    _pass(persona_dir, _self_model_only(persona_dir))
    assert stubs.order == ["self_model_articulation"]


def test_c22_code_comment_names_the_future_absorption():
    src = inspect.getsource(supervisor._run_self_model_tick).replace("#", " ")
    assert "absorbed later by the expanded central cadence function" in " ".join(src.split())


# ---------------------------------------------------------------------------
# C23 (INC-9 parts) — pass 2 is the first gated job, no interval, drains at
# every idle pass while non-empty, yields per item
# ---------------------------------------------------------------------------


def _pass2_only(persona_dir):
    return [j for j in _real_jobs(persona_dir) if j.name == "pass2"]


def _enqueue_probe(persona_dir, log, label, side=None):
    rid = pass2_queue.new_record_id()

    def _fx():
        log.append(label)
        if side is not None:
            side()

    pass2_queue.register_test_side_effect(rid, _fx)
    pass2_queue.enqueue({"id": rid, "kind": "test_probe"}, persona_dir=persona_dir)


def test_c23_pass2_drains_at_every_idle_pass_while_non_empty(tmp_path, monkeypatch):
    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    log: list[str] = []
    jobs = _pass2_only(persona_dir)

    _pass(persona_dir, jobs)  # empty queue: no work, nothing runs
    assert log == []

    _enqueue_probe(persona_dir, log, "a")
    _enqueue_probe(persona_dir, log, "b")
    _pass(persona_dir, jobs, idle=False)  # not idle: drains nothing
    assert log == [] and pass2_queue.queue_length(persona_dir) == 2

    _pass(persona_dir, jobs)
    assert log == ["a", "b"] and pass2_queue.queue_length(persona_dir) == 0
    _enqueue_probe(persona_dir, log, "c")
    _pass(persona_dir, jobs)  # the very next idle pass drains again (no interval)
    assert log == ["a", "b", "c"]
    assert not (persona_dir / "cadence").exists() or not list(
        (persona_dir / "cadence").glob("*pass2*")
    ), "pass 2 has no persisted interval (S53)"


def test_c23_pass2_job_stops_at_the_next_item_when_chat_resumes(tmp_path, monkeypatch):
    """S14 (per item): the job drains with should_pause = "chat not idle"."""
    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    log: list[str] = []
    cli_throttle.reset()
    _enqueue_probe(persona_dir, log, "a", side=lambda: cli_throttle.note_user_message())
    _enqueue_probe(persona_dir, log, "b")
    try:
        _pass(persona_dir, _pass2_only(persona_dir))
        assert log == ["a"], "a message during item a must stop the drain before item b"
        assert pass2_queue.queue_length(persona_dir) == 1
    finally:
        cli_throttle.note_reply_end()


# ---------------------------------------------------------------------------
# C15(b) — heartbeat between jobs, never during one
# ---------------------------------------------------------------------------


def test_c15b_heartbeat_due_mid_job_runs_after_the_job_and_before_the_next(tmp_path):
    events: list[str] = []
    hb = {"due": False}

    def job_a():
        events.append("A:start")
        hb["due"] = True  # the heartbeat's timer falls due while A runs
        events.append("A:end")

    def job_b():
        events.append("B:start")
        events.append("B:end")

    def between():
        if hb["due"]:
            events.append("HB:start")
            events.append("HB:end")
            hb["due"] = False

    jobs = [
        GatedJob("pass2", run=job_a, has_work=lambda: True),
        GatedJob("session_snapshot_prune", run=job_b, has_work=lambda: True),
    ]
    _pass(tmp_path, jobs, between_jobs=between)
    assert events == ["A:start", "A:end", "HB:start", "HB:end", "B:start", "B:end"]


def test_c15b_run_folded_heartbeat_and_gated_job_never_overlap(tmp_path, monkeypatch):
    """Live path: with the heartbeat due on every check (interval 0), the
    loop runs it at the top of the pass and between jobs; a gated job that is
    due starts only after the whole heartbeat pass and the next heartbeat
    starts only after the job returns (single supervisor thread, S41/S65)."""
    import threading

    persona_dir = _persona(tmp_path)
    _seed(persona_dir, "compaction_cadence.json", datetime.now(UTC) - timedelta(hours=1))
    events: list[str] = []
    stop = threading.Event()
    state = {"in_job": False, "in_hb": False}

    def fake_heartbeat(*_a, **_k):
        assert not state["in_job"], "heartbeat ran during a gated job"
        state["in_hb"] = True
        events.append("HB:start")
        events.append("HB:end")
        state["in_hb"] = False
        if events.count("C:end") >= 1 and events.count("HB:end") >= 3:
            stop.set()
        return SimpleNamespace()

    def fake_compaction(*_a, **_k):
        assert not state["in_hb"], "gated job started during a heartbeat pass"
        state["in_job"] = True
        events.append("C:start")
        events.append("C:end")
        state["in_job"] = False

    monkeypatch.setattr(supervisor, "_heartbeat_and_felt_time", fake_heartbeat)
    monkeypatch.setattr(supervisor, "_run_compaction_tick", fake_compaction)
    monkeypatch.setattr(
        "brain.chat.compaction.build_compaction_provider", lambda _pd: FakeProvider()
    )
    watchdog = threading.Timer(10.0, stop.set)
    watchdog.start()
    try:
        supervisor.run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=EventBus(),
            tick_interval_s=0.0,
            heartbeat_interval_s=0.0,
            soul_review_interval_s=None,
            finalize_interval_s=None,
            log_rotation_interval_s=None,
            initiate_review_interval_s=None,
            voice_reflection_interval_s=None,
            self_model_interval_s=None,
            calibration_interval_s=None,
            interest_sweep_interval_s=None,
            judge_selftune_interval_s=None,
            clustering_interval_s=None,
            vocab_repair_interval_s=None,
            maker_enabled=False,
            notes_enabled=False,
            kindled_link_enabled=False,
        )
    finally:
        watchdog.cancel()
    c = events.index("C:start")
    assert events[c - 2 : c] == ["HB:start", "HB:end"], events
    assert events[c + 1 : c + 4] == ["C:end", "HB:start", "HB:end"], events


# ---------------------------------------------------------------------------
# C26 — no startup call to any gated job; the one-shots left are non-gated
# ---------------------------------------------------------------------------


_GATED_CALLS = {
    "_run_compaction_tick",
    "_run_calibration_tick",
    "_run_deploy_recalibration_check",
    "_run_deploy_recalibration",
    "_emotion_backfill_run",
    "_run_clustering_tick",
    "_run_finalize_tick",
    "_run_initiate_review_tick",
    "_run_self_model_tick",
    "_run_judge_selftune_tick",
    "_embedding_backfill_run_tick",
    "snapshot_stale_sessions",
    "forgetting_run_pass",
    "run_sweep_tick",
    "drain_all_locked",
}
_NON_GATED_ONE_SHOTS = {
    "_attunement_run_backfill",
    "_attunement_run_supplementary_backfill",
    "_run_vocab_repair_tick",
    "_soul_candidate_repair_run",
    "_self_model_repair_run",
    "_delete_legacy_embeddings_db",
}


def _called_names(nodes) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    names.add(f.id)
                elif isinstance(f, ast.Attribute):
                    names.add(f.attr)
    return names


def _run_folded_prelude(source: str):
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_folded"
    )
    prelude = []
    for stmt in fn.body:
        if isinstance(stmt, ast.While):
            break
        if isinstance(stmt, ast.FunctionDef):
            continue  # nested helpers (the heartbeat hook) are not called at startup
        prelude.append(stmt)
    return prelude


def test_c26_run_folded_startup_calls_no_gated_job():
    prelude = _run_folded_prelude(Path(supervisor.__file__).read_text(encoding="utf-8"))
    called = _called_names(prelude)
    assert not (called & _GATED_CALLS), called & _GATED_CALLS
    assert called & (_NON_GATED_ONE_SHOTS | _GATED_CALLS) == _NON_GATED_ONE_SHOTS


def test_c26_scanner_positive_control_finds_a_planted_startup_call():
    src = (
        "def run_folded(stop_event):\n"
        "    _run_calibration_tick(persona_dir)\n"
        "    while not stop_event.is_set():\n"
        "        pass\n"
    )
    assert "_run_calibration_tick" in _called_names(_run_folded_prelude(src))


# ---------------------------------------------------------------------------
# C38 — deploy recalibration's due predicate (the stale-floor refit)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_reranker(monkeypatch):
    from brain.memory import reranker as reranker_mod
    from brain.memory.reranker import FakeRerankerProvider

    monkeypatch.setattr(
        reranker_mod, "build_reranker_provider", lambda **kw: FakeRerankerProvider()
    )
    return "fake-reranker"


def _floor(persona_dir, model_id, scale):
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    try:
        store.write_reranker_floor(
            model_id, floor=1.0, raw_fit_floor=1.0, sample_pairs=200,
            is_cold_start=False, score_scale=scale,
        )
    finally:
        store.close()


def test_c38_fresh_floor_is_not_due(tmp_path, fake_reranker):
    from brain.memory.store import CALIBRATION_SCORE_SCALE

    persona_dir = _persona(tmp_path)
    _floor(persona_dir, fake_reranker, CALIBRATION_SCORE_SCALE)
    assert supervisor._deploy_recalibration_due(persona_dir) is False


def test_c38_raw_scale_floor_is_due_regardless_of_retry_file(tmp_path, fake_reranker):
    persona_dir = _persona(tmp_path)
    _floor(persona_dir, fake_reranker, "raw")
    _seed(persona_dir, supervisor._DEPLOY_RECAL_RETRY_CADENCE_FILE, datetime.now(UTC) + timedelta(hours=5))
    assert supervisor._deploy_recalibration_due(persona_dir) is True


def test_c38_no_row_is_due_at_first_lull_then_retried_at_most_daily(tmp_path, fake_reranker, monkeypatch):
    from brain.memory import floor_calibration as fc_mod

    persona_dir = _persona(tmp_path)
    assert supervisor._deploy_recalibration_due(persona_dir) is True  # no retry file yet
    calls: list[str] = []
    monkeypatch.setattr(
        fc_mod,
        "derive_and_persist_floor",
        lambda store, model_id, **kw: calls.append(model_id)
        or SimpleNamespace(
            accepted=False, floor=None, is_cold_start=False,
            held_for_data_starvation=True, sample_pairs=0,
        ),
    )
    supervisor._run_deploy_recalibration(persona_dir)  # writes nothing: row still absent
    assert calls == [fake_reranker]
    assert supervisor._deploy_recalibration_due(persona_dir) is False  # retried within a day


# ---------------------------------------------------------------------------
# C39 — no-interval jobs: every idle pass while they have work, never without
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NO_INTERVAL_JOBS)
def test_c39_no_interval_job_runs_every_idle_pass_while_it_has_work(tmp_path, stubs, name):
    persona_dir = _persona(tmp_path)
    for k in stubs.work:
        stubs.work[k] = False
    jobs = [j for j in _real_jobs(persona_dir) if j.name == name]
    _pass(persona_dir, jobs)
    assert stubs.order == [], "no work → does not run"
    stubs.work[name] = True
    _pass(persona_dir, jobs)
    # INC-10: pass2's stub drain consumes its one simulated item (work[name]
    # flips False, mirroring queue_length() actually reflecting the drain —
    # see the `stubs` fixture's `_pass2_drain`), so a second pass needs a
    # freshly "enqueued" item the same way production would have one queued
    # again by the next idle lull. Every other no-interval job's has_work
    # probe is untouched by its run stub, so re-arming here is a no-op for
    # them (this line is required for pass2, harmless for the rest).
    stubs.work[name] = True
    _pass(persona_dir, jobs)
    assert stubs.order == [name, name], "has work → runs at every idle pass"
    assert not (persona_dir / "cadence").exists(), "no cadence file (S53/S66)"


def test_c39_snapshot_has_work_probe(tmp_path):
    from brain.ingest.buffer import ingest_turn, write_cursor

    persona_dir = _persona(tmp_path)
    assert supervisor._snapshot_has_work(persona_dir) is False
    sid = ingest_turn(persona_dir, {"speaker": "user", "text": "hi", "ts": "2026-09-27T10:00:00+00:00"})
    assert supervisor._snapshot_has_work(persona_dir) is True  # un-extracted turn
    write_cursor(persona_dir, sid, "2026-09-27T10:00:00+00:00")
    assert supervisor._snapshot_has_work(persona_dir) is False  # all extracted
    ingest_turn(
        persona_dir,
        {"session_id": sid, "speaker": "summary", "text": "s", "ts": "2026-09-27T11:00:00+00:00"},
    )
    assert supervisor._snapshot_has_work(persona_dir) is False  # summary blocks don't count
    ghost = persona_dir / "active_conversations" / "sess_ghost.jsonl"
    ghost.write_text("")
    assert supervisor._snapshot_has_work(persona_dir) is True  # a ghost buffer to clean


def _prune_job(persona_dir, **overrides):
    return [
        j for j in _real_jobs(persona_dir, **overrides) if j.name == "session_snapshot_prune"
    ]


def test_c39_prune_removes_a_session_that_predates_the_idle_window(tmp_path, monkeypatch):
    from brain.chat.session import all_sessions, create_session, reset_registry

    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(supervisor, "_snapshot_has_work", lambda _pd: False)
    monkeypatch.setattr(supervisor, "snapshot_stale_sessions", lambda *a, **k: [])
    reset_registry()
    try:
        sess = create_session(persona_dir.name)
        sess.created_at = datetime.now(UTC) - timedelta(seconds=3601)
        # The last message landed AFTER the session was created, a lull+ ago.
        cli_throttle.mark_interactive_active(at=__import__("time").monotonic() - 3600.0)
        _pass(persona_dir, _prune_job(persona_dir))
        assert all_sessions() == []
    finally:
        reset_registry()


def test_c39_prune_keeps_a_session_opened_during_the_idle_window(tmp_path, monkeypatch):
    from brain.chat.session import all_sessions, create_session, reset_registry

    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(supervisor, "_snapshot_has_work", lambda _pd: False)
    monkeypatch.setattr(supervisor, "snapshot_stale_sessions", lambda *a, **k: [])
    reset_registry()
    try:
        # (ii-a) bridge start, no message ever: opened AFTER this bridge
        # started, i.e. during the open window (S84).
        cli_throttle.reset()
        jobs = _prune_job(persona_dir, bridge_started_at=datetime.now(UTC) - timedelta(seconds=1))
        create_session(persona_dir.name)
        decisions = _pass(persona_dir, jobs)
        assert len(all_sessions()) == 1
        assert decisions[-1].action == "skip-not-due"  # nothing to prune → no run
        # (ii-b) a message a lull+ ago, session opened after it.
        cli_throttle.mark_interactive_active(at=__import__("time").monotonic() - 3600.0)
        _pass(persona_dir, _prune_job(persona_dir))
        assert len(all_sessions()) == 1
        # (ii-c) opened after that message but itself OLDER than the lull
        # (20 min): still inside the current idle window, so kept — a prune
        # keyed on the lull value (not the last-message anchor) would drop it.
        all_sessions()[0].created_at = datetime.now(UTC) - timedelta(minutes=20)
        _pass(persona_dir, _prune_job(persona_dir))
        assert len(all_sessions()) == 1
    finally:
        reset_registry()


def test_c39_prune_age_is_the_last_message_anchor_not_the_lull():
    src = inspect.getsource(supervisor._build_gated_jobs)
    assert "cli_throttle.time_since_last_message()" in src
    sup_src = Path(supervisor.__file__).read_text(encoding="utf-8")
    assert "chat.idle_lull_seconds" not in sup_src
    assert "_lull_seconds" not in sup_src
    assert "_SESSION_STALE_MINUTES" not in sup_src


def _mem(content="a memory long enough to embed", emotions=None):
    from brain.memory.store import Memory

    m = Memory.create_new(content=content, memory_type="meta", domain="test")
    if emotions is not None:
        m.emotions = emotions
    return m


def test_c39_emotion_backfill_has_work_probe(tmp_path):
    from brain.ingest import emotion_backfill as eb
    from brain.memory.store import MemoryStore

    persona_dir = _persona(tmp_path)
    assert eb.has_emotion_backfill_work(persona_dir) is False  # no db
    store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    try:
        store.create(_mem(emotions={"joy": 3.0}))
        assert eb.has_emotion_backfill_work(persona_dir, store=store) is False
        store.create(_mem(emotions={}))
        assert eb.has_emotion_backfill_work(persona_dir, store=store) is True
        assert eb.has_emotion_backfill_work(persona_dir) is True  # own short-lived store

        def _state(status):
            eb._save_state(  # noqa: SLF001
                persona_dir,
                eb.EmotionBackfillState(
                    started_at="x", total_memories=2, tagged_memories=1,
                    last_cursor="", status=status, schema_version="v1",
                ),
            )

        _state("complete")
        assert eb.has_emotion_backfill_work(persona_dir, store=store) is False
        # Cap hit today → no work until the budget's day is over.
        _state("deferred_to_next_day")
        now = datetime.now(UTC)
        eb._budget_path(persona_dir).write_text(  # noqa: SLF001
            json.dumps({"date": eb._today_str(now), "count": 200})  # noqa: SLF001
        )
        assert eb.has_emotion_backfill_work(persona_dir, store=store, now=now) is False
        assert eb.has_emotion_backfill_work(
            persona_dir, store=store, now=now + timedelta(days=1)
        ) is True
    finally:
        store.close()


def test_c39_embedding_backfill_has_work_probe(tmp_path):
    from brain.memory import embedding_backfill as ebf
    from brain.memory.store import MemoryStore

    persona_dir = _persona(tmp_path)
    store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    try:
        assert ebf.has_embedding_backfill_work(store) is False
        m = _mem()
        store.create(m)
        assert ebf.has_embedding_backfill_work(store) is True
        store.embed_row(m.id, m.content)
        assert ebf.has_embedding_backfill_work(store) is False
    finally:
        store.close()


# ---------------------------------------------------------------------------
# persisted_cadence.load_or_init_cadence (the S22/S69 primitive)
# ---------------------------------------------------------------------------


def test_load_or_init_cadence_present_file_is_returned_unchanged(tmp_path):
    _seed(tmp_path, "x.json", NOW - timedelta(hours=1))
    before = _cadence_path(tmp_path, "x.json").read_bytes()
    state, created = persisted_cadence.load_or_init_cadence(
        tmp_path, "x.json", now=NOW, interval_s=60.0
    )
    assert created is False and state.next_at == NOW - timedelta(hours=1)
    assert _cadence_path(tmp_path, "x.json").read_bytes() == before


def test_run_central_pass_logs_decisions(tmp_path, caplog):
    import logging

    log: list[str] = []
    with caplog.at_level(logging.INFO, logger="brain.bridge.central_cadence"):
        _pass(tmp_path, _fake_jobs(["pass2"], log))
    assert any("job=pass2 action=run" in r.getMessage() for r in caplog.records)
    assert any("job=pass2 action=completed" in r.getMessage() for r in caplog.records)


def test_paused_outcome_stops_the_sequence_without_advancing(tmp_path):
    """INC-10 seam: a job reporting PAUSED ends the sequence and its cadence
    is not advanced."""
    _seed(tmp_path, "c.json", NOW - timedelta(hours=1))
    before = _cadence_path(tmp_path, "c.json").read_bytes()
    log: list[str] = []
    jobs = [
        GatedJob(
            "compaction", run=lambda: JobOutcome.PAUSED, cadence_file="c.json", interval_s=60.0
        ),
        GatedJob("finalize", run=lambda: log.append("finalize"), has_work=lambda: True),
    ]
    decisions = _pass(tmp_path, jobs)
    assert ("compaction", "paused") in [(d.job, d.action) for d in decisions]
    assert log == []
    assert _cadence_path(tmp_path, "c.json").read_bytes() == before


def test_each_interval_cadence_file_is_read_once_per_pass(tmp_path, stubs, monkeypatch):
    persona_dir = _persona(tmp_path)
    _seed_all_overdue(persona_dir)
    calls: list[str] = []
    real = persisted_cadence.load_or_init_cadence

    def counting(pd, filename, **kw):
        calls.append(filename)
        return real(pd, filename, **kw)

    monkeypatch.setattr(persisted_cadence, "load_or_init_cadence", counting)
    _pass(persona_dir, _real_jobs(persona_dir))
    assert sorted(calls) == sorted(f for f, _i in INTERVAL_JOBS.values())


def test_pass2_reports_skipped_when_another_process_holds_the_drain_lock(tmp_path, stubs, monkeypatch):
    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(pass2_queue, "drain_all_locked", lambda *a, **k: 0)
    decisions = _pass(persona_dir, [j for j in _real_jobs(persona_dir) if j.name == "pass2"])
    assert ("pass2", "skipped") in [(d.job, d.action) for d in decisions]


# ---------------------------------------------------------------------------
# S84 (INC-9 follow-up) — no message seen in this process: the prune's idle
# window is anchored at bridge start
# ---------------------------------------------------------------------------


def _prune_setup(tmp_path, monkeypatch):
    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(supervisor, "_snapshot_has_work", lambda _pd: False)
    monkeypatch.setattr(supervisor, "snapshot_stale_sessions", lambda *a, **k: [])
    return persona_dir


def test_s84_no_message_seen_prunes_an_empty_session_older_than_bridge_start(
    tmp_path, monkeypatch
):
    """(i) No message seen (anchor -inf, nothing seeded) + an empty session
    created before this bridge started → pruned at an idle pass."""
    from brain.chat.session import all_sessions, create_session, reset_registry

    persona_dir = _prune_setup(tmp_path, monkeypatch)
    reset_registry()
    try:
        cli_throttle.reset()
        started = datetime.now(UTC)
        sess = create_session(persona_dir.name)
        sess.created_at = started - timedelta(hours=1)  # from before this bridge started
        decisions = _pass(persona_dir, _prune_job(persona_dir, bridge_started_at=started))
        assert ("session_snapshot_prune", "run") in [(d.job, d.action) for d in decisions]
        assert all_sessions() == []
    finally:
        reset_registry()


def test_s84_no_message_seen_keeps_an_empty_session_created_after_bridge_start(
    tmp_path, monkeypatch
):
    """(ii) No message seen + an empty session created after bridge start (the
    app-mount session) → kept, on this pass and on later ones (until a later
    bridge start)."""
    from brain.chat.session import all_sessions, create_session, reset_registry

    persona_dir = _prune_setup(tmp_path, monkeypatch)
    reset_registry()
    try:
        cli_throttle.reset()
        started = datetime.now(UTC) - timedelta(hours=3)
        sess = create_session(persona_dir.name)
        sess.created_at = started + timedelta(minutes=1)  # 2h59m old, but after start
        jobs = _prune_job(persona_dir, bridge_started_at=started)
        for _ in range(3):
            decisions = _pass(persona_dir, jobs)
            assert decisions[-1].action == "skip-not-due"
        assert len(all_sessions()) == 1
    finally:
        reset_registry()


def test_s84_message_seen_ignores_bridge_start(tmp_path, monkeypatch):
    """(iii) Once a message has been seen, the window opens at the last message
    as before: a session created before bridge start but AFTER the last message
    is kept; one created before the last message is pruned."""
    import time as _time

    from brain.chat.session import all_sessions, create_session, reset_registry

    persona_dir = _prune_setup(tmp_path, monkeypatch)
    reset_registry()
    try:
        started = datetime.now(UTC) - timedelta(hours=5)
        cli_throttle.mark_interactive_active(at=_time.monotonic() - 3600.0)  # 1 h ago
        keep = create_session(persona_dir.name)
        keep.created_at = datetime.now(UTC) - timedelta(minutes=30)  # after the message
        drop = create_session(persona_dir.name)
        drop.created_at = datetime.now(UTC) - timedelta(hours=2)  # before the message
        _pass(persona_dir, _prune_job(persona_dir, bridge_started_at=started))
        assert [s.session_id for s in all_sessions()] == [keep.session_id]
    finally:
        reset_registry()


def test_s84_leaves_time_since_last_message_and_is_chat_idle_untouched(tmp_path, monkeypatch):
    """The S84 fallback lives only in the prune's age: with no message seen,
    time_since_last_message() is still +inf and is_chat_idle still True (the
    S82 anchor semantics), before and after a pruning pass."""
    import math

    from brain.chat.session import create_session, reset_registry

    persona_dir = _prune_setup(tmp_path, monkeypatch)
    reset_registry()
    try:
        cli_throttle.reset()
        started = datetime.now(UTC)
        create_session(persona_dir.name).created_at = started - timedelta(hours=1)
        assert math.isinf(cli_throttle.time_since_last_message())
        _pass(persona_dir, _prune_job(persona_dir, bridge_started_at=started))
        assert math.isinf(cli_throttle.time_since_last_message())
        assert cli_throttle.is_chat_idle() is True
    finally:
        reset_registry()


def test_s84_run_folded_captures_bridge_start_before_the_loop():
    """Live wiring: run_folded records bridge start once, at entry, and hands
    it to the job table."""
    src = inspect.getsource(supervisor.run_folded)
    assert "bridge_started_at = datetime.now(UTC)" in src
    assert src.index("bridge_started_at = datetime.now(UTC)") < src.index(
        "while not stop_event.is_set():"
    )
    assert "bridge_started_at=bridge_started_at" in src


def test_s84_run_folded_live_prunes_pre_start_session_keeps_post_start_one(tmp_path):
    """Live path through a real run_folded, no message ever seen: an empty
    session that predates the supervisor's start is pruned at an idle pass; one
    opened after the start is kept."""
    import threading
    import time as _time

    from brain.chat.session import all_sessions, create_session, reset_registry

    persona_dir = _persona(tmp_path)
    reset_registry()
    cli_throttle.reset()
    old = create_session(persona_dir.name)
    old.created_at = datetime.now(UTC) - timedelta(hours=1)
    stop = threading.Event()
    t = threading.Thread(
        target=supervisor.run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": EventBus(),
            "tick_interval_s": 0.05,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_interval_s": None,
            "log_rotation_interval_s": None,
            "initiate_review_interval_s": None,
            "voice_reflection_interval_s": None,
            "self_model_interval_s": None,
            "compaction_interval_s": None,
            "calibration_interval_s": None,
            "interest_sweep_interval_s": None,
            "judge_selftune_interval_s": None,
            "clustering_interval_s": None,
            "vocab_repair_interval_s": None,
            "maker_enabled": False,
            "notes_enabled": False,
            "kindled_link_enabled": False,
        },
        daemon=True,
    )
    try:
        t.start()
        deadline = _time.monotonic() + 10.0
        while _time.monotonic() < deadline and any(
            s.session_id == old.session_id for s in all_sessions()
        ):
            _time.sleep(0.02)
        assert all(s.session_id != old.session_id for s in all_sessions()), "pre-start session kept"
        new = create_session(persona_dir.name)  # opened after bridge start
        _time.sleep(0.3)  # several 0.05 s passes
        assert any(s.session_id == new.session_id for s in all_sessions()), "post-start session pruned"
    finally:
        stop.set()
        t.join(timeout=10.0)
        reset_registry()
    assert not t.is_alive()
