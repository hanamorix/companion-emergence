"""ram-spike-fix S84 (INC-9 follow-up): the REAL wiring of "bridge start".

server.py's lifespan captures the bridge-start moment on its own thread,
BEFORE it starts the supervisor thread and before it serves any request, and
hands it to run_folded. So the app-mount session (POST /session/new, possible
as soon as the lifespan yields) is always created after bridge start and is
kept by the no-message-seen prune, however late the supervisor thread gets to
run. The supervisor here is a stand-in that never gets scheduled at all (the
worst case of the race the stage-6 red-team found): it only records its
kwargs.
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge import cli_throttle, supervisor
from brain.bridge.central_cadence import run_central_pass
from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider
from brain.bridge.server import build_app
from brain.chat.session import all_sessions, reset_registry


def test_app_mount_session_is_created_after_the_bridge_start_the_supervisor_gets(
    persona_dir: Path, monkeypatch
) -> None:
    seen: dict = {}
    released = threading.Event()

    def stalled_supervisor(stop_event, **kwargs):
        seen.update(kwargs)
        released.wait(timeout=10.0)  # never runs its own code before the request

    monkeypatch.setattr(supervisor, "run_folded", stalled_supervisor)
    cli_throttle.reset()  # no message seen in this process
    reset_registry()
    try:
        with TestClient(
            build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)
        ) as c:
            sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
            released.set()
        started = seen["bridge_started_at"]
        (sess,) = [s for s in all_sessions() if s.session_id == sid]
        assert sess.created_at > started, "the app-mount session must postdate bridge start"

        # And the prune (the real job, the real central pass) keeps it.
        monkeypatch.setattr(supervisor, "_snapshot_has_work", lambda _pd: False)
        jobs = [
            j
            for j in supervisor._build_gated_jobs(
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=EventBus(),
                is_session_busy=None,
                finalize_after_hours=24.0,
                finalize_interval_s=None,
                initiate_review_interval_s=None,
                maintenance_interval_s=None,
                self_model_interval_s=None,
                compaction_interval_s=None,
                calibration_interval_s=None,
                interest_sweep_interval_s=None,
                judge_selftune_interval_s=None,
                clustering_interval_s=None,
                intensity_drivers=lambda: None,
                tick_stats={"closed_sessions": 0, "pruned_empty_sessions": 0},
                bridge_started_at=started,
            )
            if j.name == "session_snapshot_prune"
        ]
        run_central_pass(persona_dir, jobs, is_idle=lambda: True, slot_available=lambda: True)
        assert any(s.session_id == sid for s in all_sessions())
    finally:
        released.set()
        reset_registry()
