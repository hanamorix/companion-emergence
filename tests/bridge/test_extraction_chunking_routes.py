"""#523 at the bridge: the real app's routes with stub providers.

``/sessions/snapshot`` makes one call for a 3-chunk backlog and reports ``chunks`` (response and event).
``GET /sessions/active`` with a stale 3-chunk buffer extracts every chunk in ONE request, then rolls over.
The launch rollover survives a slow or cut request and never blocks the event loop. Driven as raw ASGI on ONE event loop
(the app's lifespan run first; every ``send`` message recorded with ``time.monotonic()``; ``receive`` delivers the request then
blocks until the test injects ``http.disconnect``), because ``TestClient`` returns only the whole body.
Synthetic persona only (persona Kestrel, user Bob, made-up words).
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import brain.bridge.server as server
from brain.bridge.chat import ChatResponse
from brain.ingest import pipeline
from brain.ingest.buffer import ingest_turn, list_active_sessions, read_session
from brain.ingest.extract import format_turn_line
from brain.ingest.ingest_utils import chunk_turns

_CHUNK = 600
_TOKEN = "t0k-kestrel"
_STALE = "aaaaaaaa-1111-2222-3333-444444444444"
_FRESH = "bbbbbbbb-1111-2222-3333-444444444444"


class _Stub:
    """One provider for the whole app: extraction prompts are recorded (and may block on a per-call threading.Event);
    anything else (a compaction fold) gets a summary at once."""

    def __init__(self, *, gated: bool = False):
        self.prompts: list[str] = []
        self.gates = [threading.Event() for _ in range(16)]
        self.gated = gated
        self.on_call = None
        self._lock = threading.Lock()

    def name(self):
        return "stub"

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content="ok", tool_calls=[])

    def generate(self, prompt=None, *, system=None, **kw):
        prompt = prompt if prompt is not None else kw.get("prompt", "")
        if "extracting durable memories" not in prompt:
            return "Bob and Kestrel talked about many made-up things."
        with self._lock:
            self.prompts.append(prompt)
            n = len(self.prompts)
        if self.on_call is not None:
            self.on_call(n)
        if self.gated and not self.gates[n - 1].wait(timeout=20):
            raise RuntimeError("test gate never released")
        return "[]"


@pytest.fixture
def stub(monkeypatch):
    import brain.bridge.provider as provider_module

    s = _Stub()
    monkeypatch.setattr(provider_module, "get_provider", lambda _name, **_kw: s)
    monkeypatch.setattr(pipeline, "extraction_chunk_chars", lambda: _CHUNK, raising=False)
    return s


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    p = tmp_path / "Kestrel"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "fake", "user_name": "Bob"}))
    return p


def _seed(persona: Path, sid: str, exchanges: int, *, age_hours: float) -> list[dict]:
    start = datetime.now(UTC) - timedelta(hours=age_hours) - timedelta(seconds=exchanges * 120)
    for i in range(exchanges):
        t = start + timedelta(seconds=i * 120)
        ingest_turn(persona, {"session_id": sid, "speaker": "user", "text": f"zq{sid[:4]}u{i:03d} " + "m" * 90,
                              "ts": t.isoformat(timespec="seconds")})
        ingest_turn(persona, {"session_id": sid, "speaker": "assistant", "text": f"zq{sid[:4]}a{i:03d} " + "n" * 90,
                              "ts": (t + timedelta(seconds=1)).isoformat(timespec="seconds")})
    rows = read_session(persona, sid)
    assert len(chunk_turns(rows, max_chars=_CHUNK, user_name="Bob", assistant_name="Kestrel")) == 3
    return rows


def _line(row: dict) -> str:
    return format_turn_line(row, "Bob", "Kestrel")


# ---------------------------------------------------------------------------
# /sessions/snapshot
# ---------------------------------------------------------------------------


def test_snapshot_route_makes_one_call_and_reports_chunks(persona, stub, monkeypatch):
    _seed(persona, _FRESH, 6, age_hours=0.5)
    published: list[tuple[str, dict]] = []
    monkeypatch.setattr(server.events, "publish", lambda event_type, **payload: published.append((event_type, payload)))
    with TestClient(server.build_app(persona_dir=persona, client_origin="tests")) as c:
        r = c.post("/sessions/snapshot", json={"session_id": _FRESH})
    assert r.status_code == 200
    assert len(stub.prompts) == 1
    assert r.json()["chunks"] == 1
    events = [p for t, p in published if t == "session_snapshot"]
    assert len(events) == 1 and events[0]["chunks"] == 1


def test_the_supervisor_snapshot_job_event_carries_chunks(persona, stub):
    from brain.bridge import supervisor

    _seed(persona, _FRESH, 6, age_hours=0.5)
    published: list[dict] = []

    class _Bus:
        def publish(self, event):
            published.append(event)

    jobs = {j.name: j for j in supervisor._build_gated_jobs(**_job_table_kwargs(persona, _Bus()))}
    jobs["session_snapshot_prune"].run()
    snaps = [e for e in published if e["type"] == "session_snapshot"]
    assert len(snaps) == 1 and snaps[0]["chunks"] == 1


def _job_table_kwargs(persona: Path, bus) -> dict:
    return {
        "persona_dir": persona,
        "provider": None,
        "event_bus": bus,
        "is_session_busy": None,
        "finalize_after_hours": 24.0,
        "finalize_interval_s": None,
        "initiate_review_interval_s": None,
        "maintenance_interval_s": None,
        "self_model_interval_s": None,
        "compaction_interval_s": None,
        "calibration_interval_s": None,
        "interest_sweep_interval_s": None,
        "judge_selftune_interval_s": None,
        "clustering_interval_s": None,
        "intensity_drivers": lambda: None,
        "tick_stats": {"pruned_empty_sessions": 0, "closed_sessions": 0},
    }


# ---------------------------------------------------------------------------
# The launch rollover extracts to the end in one request (a slower launch keeps the summary)
# ---------------------------------------------------------------------------


def test_sessions_active_extracts_every_chunk_then_rolls_over(persona, stub):
    rows = _seed(persona, _STALE, 6, age_hours=30)
    present_at_call: list[bool] = []
    stub.on_call = lambda n: present_at_call.append(_STALE in list_active_sessions(persona))
    with TestClient(server.build_app(persona_dir=persona, client_origin="tests")) as c:
        r = c.get("/sessions/active")
    assert r.status_code == 200
    new_sid = r.json()["session_id"]
    assert len(stub.prompts) == 3 and present_at_call == [True, True, True]
    for row in rows:
        assert sum(_line(row) in p for p in stub.prompts) == 1
    assert new_sid not in (None, _STALE)
    assert _STALE not in list_active_sessions(persona)
    assert read_session(persona, new_sid)[0]["speaker"] == "summary"


# ---------------------------------------------------------------------------
# The launch rollover under a slow or cut request: raw-ASGI driver on one loop
# ---------------------------------------------------------------------------


class _Driver:
    def __init__(self, app):
        self.app = app
        self._life_q: asyncio.Queue | None = None
        self._life_task: asyncio.Task | None = None

    async def start(self):
        self._life_q = asyncio.Queue()
        started = asyncio.Event()

        async def receive():
            return await self._life_q.get()

        async def send(message):
            if message["type"] in ("lifespan.startup.complete", "lifespan.startup.failed"):
                started.set()

        await self._life_q.put({"type": "lifespan.startup"})
        self._life_task = asyncio.ensure_future(self.app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send))
        await asyncio.wait_for(started.wait(), 30)

    async def stop(self):
        await self._life_q.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(self._life_task, 60)

    async def get(self, path: str, *, disconnect_after_first_body: bool = False) -> list[tuple[float, dict]]:
        messages: list[tuple[float, dict]] = []
        disconnect = asyncio.Event()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append((time.monotonic(), message))
            if disconnect_after_first_body and message["type"] == "http.response.body":
                disconnect.set()

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"host", b"testserver"), (b"authorization", f"Bearer {_TOKEN}".encode())],
            "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
        }
        try:
            await self.app(scope, receive, send)
        finally:
            disconnect.set()
        return messages


def _status(messages) -> tuple[int, str]:
    start = next(m for _, m in messages if m["type"] == "http.response.start")
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    return start["status"], headers.get("content-type", "")


def _bodies(messages) -> list[tuple[float, bytes]]:
    return [(t, m.get("body", b"")) for t, m in messages if m["type"] == "http.response.body" and m.get("body")]


def _json(messages) -> dict:
    return json.loads(b"".join(b for _, b in _bodies(messages)))


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def raw_app(persona, stub, monkeypatch):
    stub.gated = True
    monkeypatch.setattr(server, "_stream_keepalive_seconds", lambda: 0.1)
    return server.build_app(persona_dir=persona, client_origin="tests", auth_token=_TOKEN)


async def _release_one_by_one(stub, *, pause: float = 0.15, wait_s: float = 10.0):
    """Release extraction calls 1-3 one at a time, ``pause`` after each arrives. Bounded: a call that never arrives (a broken
    build making fewer calls) releases every gate and returns, so the test fails instead of hanging."""
    for i in range(3):
        deadline = time.monotonic() + wait_s
        while len(stub.prompts) <= i:
            if time.monotonic() > deadline:
                for g in stub.gates:
                    g.set()
                return
            await asyncio.sleep(0.01)
        await asyncio.sleep(pause)
        stub.gates[i].set()


def test_c14a_keepalive_spaces_while_the_rollover_runs(persona, stub, raw_app):
    _seed(persona, _STALE, 6, age_hours=30)

    async def main():
        d = _Driver(raw_app)
        await d.start()
        try:
            releaser = asyncio.ensure_future(_release_one_by_one(stub))
            msgs = await d.get("/sessions/active")
            await releaser
        finally:
            await d.stop()
        return msgs

    msgs = _run(main())
    assert _status(msgs) == (200, "application/json")
    bodies = _bodies(msgs)
    # Three extraction calls are each held for `pause` (0.15 s) with a 0.1 s keepalive interval, so the stream runs for
    # more than 0.45 s: at least three empty keepalive bodies must precede the JSON. No bound on the gaps between them:
    # that is wall-clock and fails on a loaded host without a regression.
    keepalives = [b for _, b in bodies[:-1]]
    assert len(keepalives) >= 3 and all(b.strip() == b"" for b in keepalives)
    new_sid = _json(msgs)["session_id"]
    assert new_sid not in (None, _STALE) and len(stub.prompts) == 3


def test_c14a_the_fast_path_sends_the_json_first(persona, stub, monkeypatch, raw_app):
    _seed(persona, _STALE, 6, age_hours=30)
    ingest_turn(persona, {"session_id": _FRESH, "speaker": "user", "text": "hello", "ts": datetime.now(UTC).isoformat(timespec="seconds")})
    monkeypatch.setattr(server, "_stream_keepalive_seconds", lambda: 2.0)  # a long interval, so a slow host cannot fail this

    async def main():
        d = _Driver(raw_app)
        await d.start()
        try:
            t0 = time.monotonic()
            msgs = await d.get("/sessions/active")
            return msgs, t0
        finally:
            await d.stop()

    msgs, t0 = _run(main())
    bodies = _bodies(msgs)
    assert bodies[0][1].strip() == bodies[0][1] and json.loads(bodies[0][1]) == {"session_id": _FRESH}
    assert bodies[0][0] - t0 < 2.0
    assert stub.prompts == []


def test_c14b_two_concurrent_calls_share_one_rollover(persona, stub, raw_app):
    _seed(persona, _STALE, 6, age_hours=30)

    async def main():
        d = _Driver(raw_app)
        await d.start()
        try:
            releaser = asyncio.ensure_future(_release_one_by_one(stub, pause=0.1))
            a, b = await asyncio.gather(d.get("/sessions/active"), d.get("/sessions/active"))
            await releaser
        finally:
            await d.stop()
        return a, b

    a, b = _run(main())
    assert len(stub.prompts) == 3
    assert _json(a) == _json(b) and _json(a)["session_id"] not in (None, _STALE)


def test_c14c_a_cut_first_call_does_not_stop_the_rollover_and_the_replay_gets_the_new_sid(persona, stub, raw_app):
    _seed(persona, _STALE, 6, age_hours=30)
    present_at_call: list[bool] = []
    stub.on_call = lambda n: present_at_call.append(_STALE in list_active_sessions(persona))

    async def main():
        d = _Driver(raw_app)
        await d.start()
        try:
            first = await d.get("/sessions/active", disconnect_after_first_body=True)
            replay_task = asyncio.ensure_future(d.get("/sessions/active"))
            await _release_one_by_one(stub, pause=0.05)
            replay = await replay_task
        finally:
            await d.stop()
        return first, replay

    first, replay = _run(main())
    assert all(b.strip() == b"" for _, b in _bodies(first))  # cut after a keepalive space
    assert len(stub.prompts) == 3 and present_at_call == [True, True, True]
    assert _STALE not in list_active_sessions(persona)
    assert _json(replay)["session_id"] not in (None, _STALE)


@pytest.mark.parametrize("failure", ["raises", "shutdown", "refused-registration"])
def test_c14d_a_failed_rollover_answers_null(persona, stub, monkeypatch, raw_app, caplog, failure):
    from brain.bridge.cli_throttle import ShutdownDeferred

    _seed(persona, _STALE, 6, age_hours=30)
    stub.gated = False
    if failure == "raises":
        monkeypatch.setattr("brain.chat.rollover.perform_rollover", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    elif failure == "shutdown":
        monkeypatch.setattr("brain.chat.rollover.perform_rollover", lambda *a, **k: (_ for _ in ()).throw(ShutdownDeferred("stopping")))
    else:
        async def _refuse(label, fn, *a, **k):
            raise ShutdownDeferred("sealed")

        monkeypatch.setattr(server, "tracked_to_thread", _refuse)

    async def main():
        d = _Driver(raw_app)
        await d.start()
        try:
            return await d.get("/sessions/active")
        finally:
            await d.stop()

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        msgs = _run(main())
        gc.collect()
    assert _status(msgs)[0] == 200
    assert _json(msgs) == {"session_id": None}
    assert not any("exception was never retrieved" in r.getMessage() for r in caplog.records)


def _scan_blocked_until_health(monkeypatch, scan_threads: list[int], order: list[str], release, timeout_s: float):
    """Patch the buffer scan to record its thread, then block until ``release`` is set (or ``timeout_s``, a deadlock breaker
    for the on-loop variant), recording ``scan done`` in ``order``. Ordering, not wall-clock time, is the oracle."""
    import brain.ingest.buffer as buffer

    real_list = buffer.list_active_sessions

    def _blocked_list(persona_dir):
        scan_threads.append(threading.get_ident())
        release.wait(timeout=timeout_s)
        order.append("scan done")
        return real_list(persona_dir)

    monkeypatch.setattr(buffer, "list_active_sessions", _blocked_list)


def test_c14e_the_event_loop_stays_free_and_the_scan_runs_off_it(persona, stub, monkeypatch, raw_app):
    """While the launch decision's buffer scan is blocked, ``/health`` on the same loop completes; only then is the scan let go.
    So /health finishing first proves the loop was free; the scan's thread is not the loop's."""
    _seed(persona, _STALE, 6, age_hours=30)
    scan_threads: list[int] = []
    order: list[str] = []
    release = threading.Event()
    _scan_blocked_until_health(monkeypatch, scan_threads, order, release, timeout_s=30)

    async def main():
        d = _Driver(raw_app)
        await d.start()
        scan_threads.clear()  # the lifespan's own idle-anchor seed lists the buffers at startup, on the loop
        order.clear()
        loop_thread = threading.get_ident()
        try:
            active = asyncio.ensure_future(d.get("/sessions/active"))
            while not scan_threads:  # the scan has started (in its worker thread) and is now blocked
                await asyncio.sleep(0.01)
            health = await d.get("/health")
            order.append("health done")
            release.set()
            for g in stub.gates:
                g.set()
            await active
        finally:
            release.set()
            await d.stop()
        return loop_thread, health

    loop_thread, health = _run(main())
    assert _status(health)[0] == 200
    assert order == ["health done", "scan done"]
    assert scan_threads and all(t != loop_thread for t in scan_threads)


# ---- fail-first variants, run through the same driver (kept executable) ----


def test_c14_oracle_no_keepalive_variant_sends_nothing_before_the_json(persona, stub, monkeypatch, raw_app):
    _seed(persona, _STALE, 6, age_hours=30)
    monkeypatch.setattr(server, "_stream_keepalive_seconds", lambda: 3600.0)

    async def main():
        d = _Driver(raw_app)
        await d.start()
        try:
            releaser = asyncio.ensure_future(_release_one_by_one(stub))
            msgs = await d.get("/sessions/active")
            await releaser
        finally:
            await d.stop()
        return msgs

    bodies = _bodies(_run(main()))
    assert len(bodies) == 1  # (a)'s whitespace-before-the-JSON check fails on this variant


def test_c14_oracle_no_shared_task_variant_answers_null_to_the_second_call(persona, stub, monkeypatch, raw_app):
    _seed(persona, _STALE, 6, age_hours=30)

    def _always_new(s):
        return asyncio.ensure_future(
            server.tracked_to_thread("sessions active", server._sessions_active_blocking, s.persona_dir, s.persona, s.provider)
        )

    monkeypatch.setattr(server, "_shared_sessions_active_task", _always_new)

    async def main():
        d = _Driver(raw_app)
        await d.start()
        try:
            releaser = asyncio.ensure_future(_release_one_by_one(stub, pause=0.1))
            a, b = await asyncio.gather(d.get("/sessions/active"), d.get("/sessions/active"))
            await releaser
        finally:
            await d.stop()
        return a, b

    a, b = _run(main())
    assert {_json(a)["session_id"] is None, _json(b)["session_id"] is None} == {True, False}


def test_c14_oracle_on_loop_variant_blocks_health(persona, stub, monkeypatch, raw_app):
    """Fail-first kept executable: with the decision run on the loop, the blocked scan holds the loop, so /health can only finish
    after the scan (here released by its deadlock-breaking timeout), and the scan's thread is the loop's."""
    _seed(persona, _STALE, 6, age_hours=30)
    stub.gated = False
    scan_threads: list[int] = []
    order: list[str] = []
    release = threading.Event()
    _scan_blocked_until_health(monkeypatch, scan_threads, order, release, timeout_s=1.0)

    async def _on_loop(label, fn, *a, **k):
        return fn(*a, **k)

    monkeypatch.setattr(server, "tracked_to_thread", _on_loop)

    async def main():
        d = _Driver(raw_app)
        await d.start()
        scan_threads.clear()  # the lifespan's own idle-anchor seed lists the buffers at startup, on the loop
        order.clear()
        loop_thread = threading.get_ident()
        try:
            active = asyncio.ensure_future(d.get("/sessions/active"))
            await d.get("/health")
            order.append("health done")
            release.set()
            await active
        finally:
            release.set()
            await d.stop()
        return loop_thread

    loop_thread = _run(main())
    assert loop_thread in scan_threads and order == ["scan done", "health done"]


# ---------------------------------------------------------------------------
# At the route: a close that kept the buffer is not reported as a close
# ---------------------------------------------------------------------------


def test_sessions_close_with_a_late_row_answers_not_closed(persona, stub, monkeypatch):
    rows = _seed(persona, _FRESH, 6, age_hours=0.5)
    last_line = _line(rows[-1])
    published: list[str] = []
    monkeypatch.setattr(server.events, "publish", lambda event_type, **payload: published.append(event_type))

    def _late(n):
        if any(last_line in p for p in stub.prompts[n - 1 : n]):
            ingest_turn(persona, {"session_id": _FRESH, "speaker": "user", "text": "late zibbit row",
                                  "ts": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(timespec="seconds")})

    stub.on_call = _late
    with TestClient(server.build_app(persona_dir=persona, client_origin="tests")) as c:
        r = c.post("/sessions/close", json={"session_id": _FRESH})
        state = c.get(f"/state/{_FRESH}")
    assert r.status_code == 200
    assert r.json()["closed"] is False
    assert _FRESH in list_active_sessions(persona)
    assert "session_closed" not in published
    assert state.status_code == 200  # still registered
