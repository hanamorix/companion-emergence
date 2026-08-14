"""Bridge endpoint conformance tests for cascade-compaction session rollover.

Maps to changes/cascade-compaction/1.5-criteria.md:
  C8   1c-A: >24h stale-resume rollover (SYNC) + stale selection
  C9   1c-B: weekly-cap rollover driven from the DAILY TICK at a quiet moment
  C16  Post-rollover continuation redirects at the real locus (4 handlers,
       multi-generation chain, cyclic-pointer abort)
  C19  in_flight_locks keyed by the resolved sid
  C20  Close cleanup + /state use the resolved sid
  C21  Structural guard: no raw sid downstream of resolution (static check)

Drives the REAL FastAPI endpoints via TestClient and the real daily-tick
function brain.bridge.supervisor._run_compaction_tick — no bespoke helpers
standing in for the production paths.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.chat.session import get_or_hydrate_session, get_session
from brain.ingest.buffer import (
    ingest_turn,
    list_active_sessions,
    read_archive,
    read_rolled_to,
    read_session,
    write_rolled_to,
)


def _build(persona_dir: Path) -> tuple[FastAPI, TestClient]:
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return app, TestClient(app)


def _patch_fake_provider(monkeypatch, reply: str = "default reply", extraction: str = "[]"):
    """Mirrors tests/bridge/test_endpoints.py's helper — patches
    brain.bridge.server.get_provider so the lifespan's provider is a
    controllable stub. Must be called BEFORE opening the TestClient."""
    import brain.bridge.server as srv
    from brain.bridge.chat import ChatResponse

    class _Fake:
        def name(self):
            return "fake"

        def chat(self, messages, *, tools=None, options=None):
            return ChatResponse(content=reply, tool_calls=[])

        def generate(self, prompt, *, system=None):
            return extraction

    monkeypatch.setattr(srv, "get_provider", lambda _name, **_kw: _Fake())


def _make_persona(base: Path, name: str) -> Path:
    """A fresh persona dir sibling to the `persona_dir` fixture's, for
    sub-scenarios that need their own isolated app/buffer state."""
    p = base / name
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "fake"}), encoding="utf-8"
    )
    return p


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _seed(persona_dir: Path, sid: str, n: int, *, base: datetime, step: timedelta) -> None:
    for i in range(n):
        ts = _iso(base + step * i)
        speaker = "user" if i % 2 == 0 else "assistant"
        ingest_turn(
            persona_dir, {"session_id": sid, "speaker": speaker, "text": f"turn {i}", "ts": ts}
        )


class _ExtractStub:
    """generate() answers extraction prompts with a valid empty item list.

    Only used as the daily-tick's ``provider`` argument in the C9 test — the
    fold step inside perform_rollover always builds its own provider via
    build_compaction_provider (persona_config's "fake" → the real
    FakeProvider, whose non-first-person output the fold validator rejects,
    falling back to a deterministic note — the seed row still exists).
    """

    def name(self) -> str:
        return "extract-stub"

    def healthy(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        return ""

    def generate(self, prompt=None, *, system=None, **kw) -> str:
        return "[]"


# --------------------------------------------------------------------------- C8
def test_c8_idle_rollover_sync_and_selection(persona_dir: Path, tmp_path: Path, monkeypatch) -> None:
    _patch_fake_provider(monkeypatch)
    now = datetime.now(UTC)

    # (a) single stale fixture.
    sid1 = str(uuid.uuid4())
    _seed(persona_dir, sid1, 5, base=now - timedelta(hours=30), step=timedelta(minutes=1))

    app, client = _build(persona_dir)
    with client as c:
        r = c.get("/sessions/active")
        assert r.status_code == 200
        new_sid = r.json()["session_id"]
        assert new_sid is not None
        assert new_sid != sid1

    assert sid1 not in list_active_sessions(persona_dir)
    assert read_archive(persona_dir, sid1)
    new_turns = read_session(persona_dir, new_sid)
    assert new_turns
    assert new_turns[0]["speaker"] == "summary"

    # (a) H6 fail-demo: patch perform_rollover to a no-op (the pre-change
    # behavior) and confirm the endpoint DOES fall back to null.
    with monkeypatch.context() as m:
        m.setattr("brain.chat.rollover.perform_rollover", lambda *a, **kw: None)
        persona_dir_fd = _make_persona(tmp_path, "c8-fail-demo")
        _seed(persona_dir_fd, str(uuid.uuid4()), 3, base=now - timedelta(hours=30), step=timedelta(minutes=1))
        app_fd, client_fd = _build(persona_dir_fd)
        with client_fd as c:
            r_fd = c.get("/sessions/active")
            assert r_fd.status_code == 200
            assert r_fd.json()["session_id"] is None

    # (b) multi-stale: the MOST-RECENTLY-ACTIVE stale buffer is the one rolled.
    persona_dir_b = _make_persona(tmp_path, "c8-multi-stale")
    sid_x, sid_y = str(uuid.uuid4()), str(uuid.uuid4())
    # Assign so the lexicographically-larger sid is the OLDER (staler) one —
    # an "arbitrary"/lexicographic-max pick would choose wrong.
    recent_sid, older_sid = (sid_x, sid_y) if sid_x < sid_y else (sid_y, sid_x)
    naive_pick = max(older_sid, recent_sid)
    assert naive_pick == older_sid  # sanity: fixture IS representative of the bug

    _seed(persona_dir_b, older_sid, 3, base=now - timedelta(hours=48), step=timedelta(minutes=1))
    _seed(persona_dir_b, recent_sid, 3, base=now - timedelta(hours=30), step=timedelta(minutes=1))

    app_b, client_b = _build(persona_dir_b)
    with client_b as c:
        r_b = c.get("/sessions/active")
        assert r_b.status_code == 200
        picked_new_sid = r_b.json()["session_id"]
    assert picked_new_sid is not None
    assert read_rolled_to(persona_dir_b, recent_sid) == picked_new_sid
    assert read_rolled_to(persona_dir_b, older_sid) is None
    assert older_sid in list_active_sessions(persona_dir_b)


# --------------------------------------------------------------------------- C9
def test_c9_weekly_rollover_daily_tick(persona_dir: Path, monkeypatch) -> None:
    from brain.bridge.supervisor import _run_compaction_tick

    now = datetime.now(UTC)
    sid = str(uuid.uuid4())
    provider = _ExtractStub()

    idx = 0

    def _put(pdir: Path, s: str, ts: datetime) -> None:
        nonlocal idx
        speaker = "user" if idx % 2 == 0 else "assistant"
        ingest_turn(pdir, {"session_id": s, "speaker": speaker, "text": f"turn {idx}", "ts": _iso(ts)})
        idx += 1

    # 5 turns >72h old, 5 in (48h,72h], 5 in (24h,48h] — populates all 3 tiers.
    for k in range(5):
        _put(persona_dir, sid, now - timedelta(days=8) + timedelta(minutes=k))
    for k in range(5):
        _put(persona_dir, sid, now - timedelta(hours=60) + timedelta(minutes=k))
    for k in range(5):
        _put(persona_dir, sid, now - timedelta(hours=30) + timedelta(minutes=k))
    # Protected 40-message tail, ending ~35 minutes ago (past the 30-min quiet gap).
    tail_start = now - timedelta(minutes=35 + 39)
    for k in range(40):
        _put(persona_dir, sid, tail_start + timedelta(minutes=k))
    assert idx == 55

    _run_compaction_tick(persona_dir, provider)

    new_sid = read_rolled_to(persona_dir, sid)
    assert new_sid is not None  # (a) the daily tick fired the swap

    new_turns = read_session(persona_dir, new_sid)
    summary_rows = [t for t in new_turns if t["speaker"] == "summary"]
    raw_rows = [t for t in new_turns if t["speaker"] != "summary"]
    assert len(summary_rows) == 1
    sections = summary_rows[0]["compaction"]["sections"]
    assert set(sections) == {"24h", "48h", "72h"}  # (b) 3 tiers
    assert len(raw_rows) == 40  # (b) + the 40 most-recent messages

    assert sid not in list_active_sessions(persona_dir)  # (c) old buffer deleted
    assert read_archive(persona_dir, sid)  # (c) archived

    # (d) a fixture with a RECENT last turn (within the quiet gap) → defers,
    # no mid-exchange fire.
    sid_recent = str(uuid.uuid4())
    idx = 0

    def _put2(ts: datetime) -> None:
        nonlocal idx
        speaker = "user" if idx % 2 == 0 else "assistant"
        ingest_turn(
            persona_dir,
            {"session_id": sid_recent, "speaker": speaker, "text": f"r{idx}", "ts": _iso(ts)},
        )
        idx += 1

    _put2(now - timedelta(days=8))
    _put2(now - timedelta(minutes=5))  # last turn only 5 min ago — inside the 30-min quiet gap

    _run_compaction_tick(persona_dir, provider)
    assert read_rolled_to(persona_dir, sid_recent) is None
    assert sid_recent in list_active_sessions(persona_dir)

    # H6 fail-demo: force quiet_gap=0 and confirm the SAME recent-last-turn
    # fixture WOULD fire a mid-exchange swap — proving it's the real
    # quiet-gap gate, not luck, that deferred it above.
    import brain.chat.compaction as compaction_mod

    monkeypatch.setattr(compaction_mod, "_ROLLOVER_QUIET_GAP", timedelta(0))
    _run_compaction_tick(persona_dir, provider)
    assert read_rolled_to(persona_dir, sid_recent) is not None


# --------------------------------------------------------------------------- C16
def test_c16_post_rollover_continuation_redirect(persona_dir: Path, monkeypatch) -> None:
    _patch_fake_provider(monkeypatch)
    persona_name = persona_dir.name
    now = datetime.now(UTC)

    sid1 = str(uuid.uuid4())
    sid2 = str(uuid.uuid4())
    _seed(persona_dir, sid2, 2, base=now - timedelta(minutes=10), step=timedelta(minutes=1))
    write_rolled_to(persona_dir, sid1, sid2)

    # (a) the real resolution chokepoint redirects the old sid.
    sess = get_or_hydrate_session(persona_dir, persona_name, sid1)
    assert sess is not None
    assert sess.session_id == sid2

    app, client = _build(persona_dir)
    with client as c:
        # (b) POST /chat with the OLD sid -> 200, turn lands in the SUCCESSOR buffer.
        before_len = len(read_session(persona_dir, sid2))
        r = c.post("/chat", json={"session_id": sid1, "message": "hello"})
        assert r.status_code == 200
        assert r.json()["session_id"] == sid2
        assert len(read_session(persona_dir, sid2)) == before_len + 2  # user + assistant

        # /sessions/snapshot with the OLD sid -> operates on the successor.
        r2 = c.post("/sessions/snapshot", json={"session_id": sid1})
        assert r2.status_code == 200
        assert r2.json()["session_id"] == sid2
        assert r2.json()["closed"] is False

        # /sessions/close with the OLD sid -> operates on the successor, not a
        # false committed=0 no-op / not a 404.
        r3 = c.post("/sessions/close", json={"session_id": sid1})
        assert r3.status_code == 200
        body3 = r3.json()
        assert body3["session_id"] == sid2
        assert body3["closed"] is True

    assert get_session(sid2) is None
    assert sid2 not in list_active_sessions(persona_dir)


def test_c16_multi_generation_and_cyclic_redirect(persona_dir: Path, tmp_path: Path, monkeypatch) -> None:
    _patch_fake_provider(monkeypatch)
    now = datetime.now(UTC)

    # Multi-generation: sid1 -> sid2 -> sid3 (three successive rollovers).
    persona_dir_mg = _make_persona(tmp_path, "c16-multi-gen")
    persona_name = persona_dir_mg.name
    sid1, sid2, sid3 = (str(uuid.uuid4()) for _ in range(3))
    _seed(persona_dir_mg, sid3, 2, base=now - timedelta(minutes=5), step=timedelta(minutes=1))
    write_rolled_to(persona_dir_mg, sid1, sid2)
    write_rolled_to(persona_dir_mg, sid2, sid3)

    sess = get_or_hydrate_session(persona_dir_mg, persona_name, sid1)
    assert sess is not None
    assert sess.session_id == sid3  # full chain follow, not single-hop

    app_mg, client_mg = _build(persona_dir_mg)
    with client_mg as c:
        before = len(read_session(persona_dir_mg, sid3))
        r = c.post("/chat", json={"session_id": sid1, "message": "hi"})
        assert r.status_code == 200
        assert r.json()["session_id"] == sid3
        assert len(read_session(persona_dir_mg, sid3)) == before + 2

    # Cyclic pointer (corrupt) aborts to None / 404, not a hang.
    persona_dir_cyc = _make_persona(tmp_path, "c16-cyclic")
    cyc1, cyc2 = str(uuid.uuid4()), str(uuid.uuid4())
    write_rolled_to(persona_dir_cyc, cyc1, cyc2)
    write_rolled_to(persona_dir_cyc, cyc2, cyc1)
    assert get_or_hydrate_session(persona_dir_cyc, "cyclic", cyc1) is None

    app_cyc, client_cyc = _build(persona_dir_cyc)
    with client_cyc as c:
        r_cyc = c.get(f"/state/{cyc1}")
        assert r_cyc.status_code == 404


# --------------------------------------------------------------------------- C19
def test_c19_inflight_lock_keyed_by_resolved_sid(persona_dir: Path, monkeypatch) -> None:
    _patch_fake_provider(monkeypatch)
    now = datetime.now(UTC)
    sid1 = str(uuid.uuid4())
    sid2 = str(uuid.uuid4())
    _seed(persona_dir, sid2, 2, base=now - timedelta(minutes=10), step=timedelta(minutes=1))
    write_rolled_to(persona_dir, sid1, sid2)

    app, client = _build(persona_dir)
    with client as c:
        state = app.state.bridge

        r1 = c.post("/chat", json={"session_id": sid1, "message": "hi"})
        assert r1.status_code == 200
        assert set(state.in_flight_locks.keys()) == {sid2}
        assert sid1 not in state.in_flight_locks

        r2 = c.post("/chat", json={"session_id": sid2, "message": "hi again"})
        assert r2.status_code == 200
        assert set(state.in_flight_locks.keys()) == {sid2}  # still one key, not two

        # Discriminating check: manually hold the SUCCESSOR's lock (simulating
        # a genuinely in-flight turn) and confirm traffic on the OLD sid
        # serialises on that SAME key (429), not a fresh unlocked lock under
        # a different (raw-old-sid) key.
        lock = state.in_flight_locks[sid2]
        asyncio.run(lock.acquire())
        try:
            r3 = c.post("/chat", json={"session_id": sid1, "message": "should be busy"})
            assert r3.status_code == 429
        finally:
            lock.release()
        assert sid1 not in state.in_flight_locks


# --------------------------------------------------------------------------- C20
def test_c20_close_cleanup_uses_resolved_sid(persona_dir: Path, monkeypatch) -> None:
    _patch_fake_provider(monkeypatch)
    persona_name = persona_dir.name
    now = datetime.now(UTC)
    sid1 = str(uuid.uuid4())
    sid2 = str(uuid.uuid4())
    _seed(persona_dir, sid2, 2, base=now - timedelta(minutes=10), step=timedelta(minutes=1))
    write_rolled_to(persona_dir, sid1, sid2)

    app, client = _build(persona_dir)
    with client as c:
        state = app.state.bridge

        # /state's in-flight lookup uses the resolved sid: mark the
        # SUCCESSOR's lock as held and confirm /state on the OLD sid reports it.
        state.in_flight_locks[sid2] = asyncio.Lock()
        asyncio.run(state.in_flight_locks[sid2].acquire())
        r_state = c.get(f"/state/{sid1}")
        assert r_state.status_code == 200
        body = r_state.json()
        assert body["session_id"] == sid2
        assert body["in_flight"] is True
        state.in_flight_locks[sid2].release()

        # Hydrate sid2 into the registry so its removal is observable.
        assert get_or_hydrate_session(persona_dir, persona_name, sid2) is not None
        assert get_session(sid2) is not None

        # Close via the OLD sid -> removes the SUCCESSOR's registry entry and
        # pops the SUCCESSOR's lock key (not a raw-old-sid no-op).
        r_close = c.post("/sessions/close", json={"session_id": sid1})
        assert r_close.status_code == 200
        assert r_close.json()["session_id"] == sid2

        assert get_session(sid2) is None
        assert sid2 not in state.in_flight_locks

        # After the successor is fully drained, /state on the old sid reports
        # the TRUE (gone) state — 404, not a stale/false success.
        r_state2 = c.get(f"/state/{sid1}")
        assert r_state2.status_code == 404


# --------------------------------------------------------------------------- C21
_C21_HANDLER_NAMES = ("chat", "stream", "sessions_snapshot", "sessions_close", "state_endpoint")


def _handler_blocks(source: str) -> dict[str, str]:
    """Split server.py's build_app body into per-endpoint-handler source
    slices, keyed by function name. Each slice runs from one top-level
    ``    @app.`` decorator up to (not including) the next one."""
    starts = [m.start() for m in re.finditer(r"\n    @app\.", source)]
    starts.append(len(source))
    blocks: dict[str, str] = {}
    for i in range(len(starts) - 1):
        chunk = source[starts[i] : starts[i + 1]]
        m = re.search(r"\n\s*(?:async )?def (\w+)\(", chunk)
        if m:
            blocks.setdefault(m.group(1), chunk)
    return blocks


def _rebind_and_loadbearing_ok(block: str) -> tuple[bool, str]:
    """(ok, reason). ok=True iff a resolved-sid rebind (``sid = sess.session_id``
    or ``session_id = sess.session_id``) precedes every load-bearing use of
    in_flight_locks / remove_session in the block, and none of those
    load-bearing lines reference the raw ``req.session_id``."""
    if "get_or_hydrate_session(" not in block:
        return False, "no get_or_hydrate_session call"
    lines = block.splitlines()
    rebind_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*(sid|session_id)\s*=\s*sess\.session_id\s*$", line):
            rebind_idx = i
            break
    if rebind_idx is None:
        return False, "no resolved-sid rebind"
    marker_re = re.compile(
        r"in_flight_locks\.setdefault\(|in_flight_locks\.pop\(|remove_session\(|"
        r"session_id\s+in\s+s\.in_flight_locks"
    )
    for i, line in enumerate(lines):
        if marker_re.search(line):
            if i < rebind_idx:
                return False, f"load-bearing use before rebind: {line.strip()!r}"
            if "req.session_id" in line:
                return False, f"load-bearing use references req.session_id: {line.strip()!r}"
    return True, "ok"


def test_c21_no_raw_sid_downstream_of_resolution() -> None:
    server_path = Path(__file__).resolve().parents[2] / "brain" / "bridge" / "server.py"
    source = server_path.read_text(encoding="utf-8")
    blocks = _handler_blocks(source)
    for name in _C21_HANDLER_NAMES:
        assert name in blocks, f"handler {name!r} not found in server.py"
        ok, reason = _rebind_and_loadbearing_ok(blocks[name])
        assert ok, f"{name}: {reason}"


def test_c21_flags_all_five_sites_on_pre_fix_base_commit() -> None:
    """H6: run the SAME structural check against the base commit
    (cd29bc61, before the redirect work landed) and confirm it flags EVERY
    one of the 5 sites — the current tree passing this check is not vacuous."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "show", "cd29bc61:brain/bridge/server.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    base_source = result.stdout
    blocks = _handler_blocks(base_source)
    failing = []
    for name in _C21_HANDLER_NAMES:
        block = blocks.get(name)
        if block is None:
            failing.append(name)
            continue
        ok, _reason = _rebind_and_loadbearing_ok(block)
        if not ok:
            failing.append(name)
    assert set(failing) == set(_C21_HANDLER_NAMES), (
        f"expected the check to flag ALL 5 pre-fix sites, only flagged {failing}"
    )


def test_owner_idle_gate_defers_busy_session(persona_dir: Path) -> None:
    """Owner idle-gate (2026-08-13): the supervisor compaction tick fires ONLY at
    startup or during idle — never mid-exchange. A session with an in-flight request
    (``is_session_busy(sid)`` true) is SKIPPED entirely — neither its cascade nor its
    weekly rollover runs — so the rollover can never delete an active session's buffer
    out from under a mid-tool-loop request (the resolve-persist race). When idle (==
    startup, which is idle by nature) the same tick fires normally.

    Shown-able-to-fail: without the idle-gate, the BUSY tick would roll the session
    (the pre-gate behaviour) — the ``read_rolled_to(...) is None`` assertion on the
    busy tick is the discriminating guard.
    """
    from brain.bridge.supervisor import _run_compaction_tick
    from brain.chat.session import reset_registry

    now = datetime.now(UTC)
    sid = str(uuid.uuid4())
    provider = _ExtractStub()

    idx = 0

    def _put(ts: datetime) -> None:
        nonlocal idx
        speaker = "user" if idx % 2 == 0 else "assistant"
        ingest_turn(
            persona_dir,
            {"session_id": sid, "speaker": speaker, "text": f"turn {idx}", "ts": _iso(ts)},
        )
        idx += 1

    # Weekly-eligible: ≥7d old, all 3 age bands populated, last turn ~35min ago
    # (past the 30-min quiet gap → user-quiet). Same shape as the C9 fixture.
    for k in range(5):
        _put(now - timedelta(days=8) + timedelta(minutes=k))
    for k in range(5):
        _put(now - timedelta(hours=60) + timedelta(minutes=k))
    for k in range(5):
        _put(now - timedelta(hours=30) + timedelta(minutes=k))
    tail_start = now - timedelta(minutes=35 + 39)
    for k in range(40):
        _put(tail_start + timedelta(minutes=k))

    reset_registry()
    try:
        # BUSY → the whole session is skipped: no rollover, still active, and NO
        # sectioned summary was written (the cascade was skipped too).
        _run_compaction_tick(persona_dir, provider, is_session_busy=lambda _s: True)
        assert read_rolled_to(persona_dir, sid) is None
        assert sid in list_active_sessions(persona_dir)
        rows = read_session(persona_dir, sid)
        assert not [r for r in rows if r.get("speaker") == "summary"], (
            "busy session must not be compacted mid-request"
        )

        # IDLE (== the startup catch-up, which passes a busy-check that returns
        # False) → the weekly rollover fires.
        _run_compaction_tick(persona_dir, provider, is_session_busy=lambda _s: False)
        assert read_rolled_to(persona_dir, sid) is not None
        assert sid not in list_active_sessions(persona_dir)
    finally:
        reset_registry()


def test_weekly_rollover_belt_defers_when_busy(persona_dir: Path) -> None:
    """Unit belt for the idle-gate: maybe_weekly_rollover returns None (defer) when
    ``is_session_busy`` reports an in-flight request, even for an otherwise-eligible
    (aged + quiet) session — the direct guard on the buffer-deleting path."""
    from brain.chat.rollover import maybe_weekly_rollover

    now = datetime.now(UTC)
    sid = str(uuid.uuid4())
    # Aged ≥7d, last turn ~1d ago (well past the quiet gap).
    _seed(persona_dir, sid, 6, base=now - timedelta(days=8), step=timedelta(hours=1))

    got = maybe_weekly_rollover(
        persona_dir, sid, "persona",
        weekly_age=timedelta(days=7), quiet_gap=timedelta(minutes=30),
        now=now, is_session_busy=lambda _s: True,
    )
    assert got is None, "must defer while a request is in-flight"
    assert read_rolled_to(persona_dir, sid) is None
