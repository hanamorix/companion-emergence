"""The central cadence pass and a refused Claude call (C2, C38 a-c, f; D13).

A refusal ("not now") never advances a job's schedule. Only the shutdown flag (or ``should_stop``) ends the pass; an
expired-login refusal backs the job off for the login re-probe interval, in memory; a lost slot race skips the job for
this pass only. Any other exception still advances the schedule (S44)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge import central_cadence, cli_throttle, provider_auth
from brain.bridge.central_cadence import GatedJob, JobOutcome

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean():
    cli_throttle.reset()
    provider_auth.reset()
    yield
    cli_throttle.reset()
    provider_auth.reset()


def _cadence_job(name: str, ran: list[str], raises: BaseException | None = None) -> GatedJob:
    def run():
        ran.append(name)
        if raises is not None:
            raise raises
        return JobOutcome.COMPLETED

    return GatedJob(name, run=run, cadence_file=f"{name}_cadence.json", interval_s=3600.0)


def _work_job(name: str, ran: list[str], raises: BaseException | None = None) -> GatedJob:
    def run():
        ran.append(name)
        if raises is not None:
            raise raises
        return JobOutcome.COMPLETED

    return GatedJob(name, run=run, has_work=lambda: True)


def _due(persona_dir: Path, name: str) -> Path:
    from brain.paths import cadence_state_path

    path = cadence_state_path(persona_dir, f"{name}_cadence.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"next_at": "2026-01-01T00:00:00+00:00"}', encoding="utf-8")
    return path


def _pass(persona_dir, jobs, **kw):
    return central_cadence.run_central_pass(
        persona_dir, jobs, is_idle=lambda: True, slot_available=lambda: True, now_wall=lambda: NOW, **kw
    )


def test_should_stop_ends_the_pass_between_jobs(tmp_path):
    ran: list[str] = []
    files = {n: _due(tmp_path, n) for n in ("maintenance", "interest_sweep")}
    before = {n: f.read_bytes() for n, f in files.items()}
    stop = {"now": False}

    def first():
        ran.append("maintenance")
        stop["now"] = True
        return JobOutcome.COMPLETED  # a job that finished; the stop arrived meanwhile

    jobs = [
        GatedJob("maintenance", run=first, cadence_file="maintenance_cadence.json", interval_s=3600.0),
        _cadence_job("interest_sweep", ran),
    ]
    _pass(tmp_path, jobs, should_stop=lambda: stop["now"])
    assert ran == ["maintenance"]
    assert files["interest_sweep"].read_bytes() == before["interest_sweep"]


def test_should_stop_before_the_first_job_runs_nothing(tmp_path):
    ran: list[str] = []
    _due(tmp_path, "maintenance")
    _pass(tmp_path, [_cadence_job("maintenance", ran)], should_stop=lambda: True)
    assert ran == []


def test_a_lost_slot_refusal_skips_the_job_and_the_pass_continues(tmp_path):
    ran: list[str] = []
    files = {n: _due(tmp_path, n) for n in ("maintenance", "interest_sweep")}
    before = files["maintenance"].read_bytes()
    jobs = [
        _cadence_job("maintenance", ran, raises=cli_throttle.ThrottleDeferred("slot lost")),
        _cadence_job("interest_sweep", ran),
        _work_job("cost_series_flush", ran),
    ]
    decisions = _pass(tmp_path, jobs)
    assert ran == ["cost_series_flush", "maintenance", "interest_sweep"]
    assert files["maintenance"].read_bytes() == before  # (a) no advance for the refused job
    assert files["interest_sweep"].read_bytes() != before  # the next job ran and advanced
    assert ("maintenance", "skip-deferred") in {(d.job, d.action) for d in decisions}


def test_a_shutdown_refusal_ends_the_pass(tmp_path):
    ran: list[str] = []
    files = {n: _due(tmp_path, n) for n in ("maintenance", "interest_sweep")}
    before = {n: f.read_bytes() for n, f in files.items()}

    def refuse():
        ran.append("maintenance")
        cli_throttle.request_shutdown()
        raise cli_throttle.ShutdownDeferred("stopping")

    jobs = [
        GatedJob("maintenance", run=refuse, cadence_file="maintenance_cadence.json", interval_s=3600.0),
        _cadence_job("interest_sweep", ran),
    ]
    _pass(tmp_path, jobs)
    assert ran == ["maintenance"]  # (b)
    assert {n: f.read_bytes() for n, f in files.items()} == before


def test_any_other_exception_still_advances(tmp_path):
    ran: list[str] = []
    f = _due(tmp_path, "maintenance")
    before = f.read_bytes()
    _pass(tmp_path, [_cadence_job("maintenance", ran, raises=RuntimeError("boom"))])
    assert f.read_bytes() != before  # (c) S44 kept


def test_expired_login_refusal_backs_the_job_off_for_the_reprobe_interval(tmp_path, monkeypatch):
    """(f): during an expired login a refused job runs at most once per re-probe interval, its schedule unchanged,
    the other jobs keep running, and it runs again on the next pass once the login is back."""
    ran: list[str] = []
    f = _due(tmp_path, "daily_calibration")
    before = f.read_bytes()
    clock = {"t": 1000.0}
    monkeypatch.setattr(central_cadence.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(provider_auth, "login_expired", lambda: True)
    deferred: dict[str, float] = {}
    jobs = [
        _cadence_job("daily_calibration", ran, raises=provider_auth.ProviderAuthDeferred("expired")),
        _work_job("finalize", ran),
    ]
    interval = provider_auth.reprobe_interval_s()
    for _ in range(10):  # ten passes inside one interval
        _pass(tmp_path, jobs, deferred_until=deferred)
        clock["t"] += interval / 20
    assert ran.count("daily_calibration") == 1
    assert ran.count("finalize") == 10
    assert f.read_bytes() == before
    clock["t"] += interval  # past the interval: tried once more
    _pass(tmp_path, jobs, deferred_until=deferred)
    assert ran.count("daily_calibration") == 2
    # login back: the job runs on the very next pass
    monkeypatch.setattr(provider_auth, "login_expired", lambda: False)
    ok_jobs = [_cadence_job("daily_calibration", ran), _work_job("finalize", ran)]
    _pass(tmp_path, ok_jobs, deferred_until=deferred)
    assert ran.count("daily_calibration") == 3


@pytest.mark.parametrize("refusal", ["auth", "shutdown"])
def test_snapshot_refusal_still_runs_the_prune_half_once(tmp_path, monkeypatch, refusal):
    """C38 (e): the empty-session prune is indivisible and runs once per pass even when the snapshot half was
    refused; the refusal still reaches the central pass (no completion recorded)."""
    import brain.bridge.supervisor as sup
    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider

    pd = tmp_path / "nell"
    (pd / "active_conversations").mkdir(parents=True)
    exc = (
        cli_throttle.ShutdownDeferred("stopping") if refusal == "shutdown"
        else provider_auth.ProviderAuthDeferred("expired")
    )

    def refuse(*_a, **_k):
        raise exc

    pruned: list[int] = []
    monkeypatch.setattr(sup, "snapshot_stale_sessions", refuse)
    monkeypatch.setattr(sup, "prune_empty_sessions", lambda **_k: pruned.append(1) or [])
    monkeypatch.setattr(sup, "build_tier_provider", lambda *_a, **_k: FakeProvider())
    jobs = sup._build_gated_jobs(
        persona_dir=pd, provider=FakeProvider(), event_bus=EventBus(), is_session_busy=None,
        finalize_after_hours=24.0, finalize_interval_s=None, initiate_review_interval_s=None,
        maintenance_interval_s=None, self_model_interval_s=None, compaction_interval_s=None,
        calibration_interval_s=None, interest_sweep_interval_s=None, judge_selftune_interval_s=None,
        clustering_interval_s=None, intensity_drivers=lambda: None,
        tick_stats={"closed_sessions": 0, "pruned_empty_sessions": 0},
    )
    job = next(j for j in jobs if j.name == "session_snapshot_prune")
    with pytest.raises(type(exc)):
        job.run()
    assert pruned == [1]
