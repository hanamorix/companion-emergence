# SP-7 Bridge Daemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-persona FastAPI daemon at `brain/bridge/` that wraps the SP-6 chat engine, exposes it over HTTP + WebSocket on `127.0.0.1`, folds the conversation supervisor as an in-process thread, broadcasts brain events, and recovers cleanly from dirty shutdowns.

**Architecture:** One daemon process per persona. Dynamic port written to `<persona_dir>/bridge.json` (with `shutdown_clean` flag for crash recovery). Streaming is *simulated* (Claude CLI returns full reply, server chunks word-by-word). Supervisor is a non-daemon thread running `close_stale_sessions` every 60s. Engines publish events via a thread-safe `EventBus` with drop-on-overflow queues.

**Tech Stack:** Python 3.12, FastAPI + uvicorn (new deps), starlette TestClient for sync tests, existing `brain/chat/engine.py` (SP-6), `brain/ingest/pipeline.py` (SP-4), `brain/health/attempt_heal.save_with_backup` for atomic state writes.

**Spec:** `docs/superpowers/specs/2026-04-28-sp-7-bridge-daemon-design.md`

**Smoke-test discipline:** Every task ends with a smoke-test step that *runs the actual code* and observes output before commit. This is a hard gate, not an aside — Hana flagged it explicitly.

---

## File Structure

**New files:**

| File | Purpose |
|---|---|
| `brain/bridge/events.py` | Module-level `set_publisher` / `publish` + `EventBus` class |
| `brain/bridge/state_file.py` | `bridge.json` schema, atomic write, dirty-shutdown predicates, pid helpers |
| `brain/bridge/server.py` | FastAPI app, route handlers, lifespan, `/health` + `/session/new` + `/state/{sid}` + `/chat` + `/stream` + `/events` + `/sessions/close` |
| `brain/bridge/supervisor.py` | `run_folded(stop_event, ...)` — non-daemon thread, 60s tick |
| `brain/bridge/daemon.py` | Process spawn/stop/status, dirty-shutdown recovery orchestration, CLI handlers |
| `tests/bridge/__init__.py` | Empty package init |
| `tests/bridge/conftest.py` | Shared fixtures: `persona_dir`, `singletons`, `bridge_app` |
| `tests/bridge/test_events.py` | 6 unit tests |
| `tests/bridge/test_state_file.py` | 4 unit tests |
| `tests/bridge/test_endpoints.py` | 6 integration tests against `TestClient` |
| `tests/bridge/test_event_stream.py` | 5 integration tests for `/events` WS |
| `tests/bridge/test_lifecycle.py` | 5 integration tests for shutdown / recovery / idle |

**Modified files:**

| File | Change |
|---|---|
| `pyproject.toml` | Add `fastapi`, `uvicorn[standard]` to dependencies |
| `brain/cli.py` | Add `nell bridge {start,stop,status,tail-events}` subcommand group; add `--no-bridge`/`--bridge-only` flags + auto-spawn-and-route to `nell chat` |

---

## Task 1: Dependencies + package scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `brain/bridge/__init__.py` (already exists per spec — verify and reuse)
- Create: `tests/bridge/__init__.py`
- Create: `tests/bridge/conftest.py`

- [ ] **Step 1: Add FastAPI + uvicorn to pyproject.toml**

Edit `pyproject.toml`, the `dependencies` list, to add two lines:

```toml
dependencies = [
    "platformdirs>=4.2",
    "numpy>=1.26",
    "ddgs>=6.0",
    "httpx>=0.27",
    "mcp>=1.0.0,<2.0.0",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
]
```

`uvicorn[standard]` pulls in `websockets`, `httptools`, and `uvloop` — needed for the WS endpoints and faster event-loop performance.

- [ ] **Step 2: Install the deps**

```bash
cd /Users/hanamori/companion-emergence
uv sync
```

Expected: lockfile updates, no errors.

- [ ] **Step 3: Smoke-test the install**

```bash
uv run python -c "import fastapi, uvicorn, websockets; print(fastapi.__version__, uvicorn.__version__)"
```

Expected: prints two version strings, no `ImportError`.

- [ ] **Step 4: Create `tests/bridge/__init__.py` (empty)**

```bash
mkdir -p tests/bridge
touch tests/bridge/__init__.py
```

- [ ] **Step 5: Create `tests/bridge/conftest.py` with shared fixtures**

```python
"""Shared fixtures for bridge tests."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    """A fresh empty persona directory for one test."""
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


@pytest.fixture
def store(persona_dir: Path) -> Iterator[MemoryStore]:
    """An open MemoryStore against an in-tmp SQLite db."""
    s = MemoryStore(persona_dir / "memories.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def hebbian(persona_dir: Path) -> Iterator[HebbianMatrix]:
    """An open HebbianMatrix against an in-tmp SQLite db."""
    h = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        yield h
    finally:
        h.close()


@pytest.fixture
def embeddings() -> EmbeddingCache:
    """A fresh in-memory embedding cache."""
    return EmbeddingCache()
```

- [ ] **Step 6: Verify the test directory boots**

```bash
uv run pytest tests/bridge/ --collect-only -q
```

Expected: `0 tests collected`, no import errors. (Tests come in later tasks.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/bridge/__init__.py tests/bridge/conftest.py
git commit -m "chore(sp7): add FastAPI/uvicorn deps + bridge test scaffolding"
```

---

## Task 2: Event bus module

**Files:**
- Create: `brain/bridge/events.py`
- Create: `tests/bridge/test_events.py`

The event bus has two layers: a module-level `publish()` engines call (no-op when no bridge) and an `EventBus` class the bridge constructs at startup. `EventBus.publish` is thread-safe via `call_soon_threadsafe`; queues are bounded with drop-oldest semantics.

- [ ] **Step 1: Write the failing test for module-level `publish` no-op**

Create `tests/bridge/test_events.py`:

```python
"""Bridge event bus — module-level publisher + EventBus class."""
from __future__ import annotations

import asyncio

import pytest

from brain.bridge import events


def test_publish_is_noop_when_no_publisher_registered():
    """When set_publisher hasn't been called, publish() returns silently."""
    events.set_publisher(None)
    # Must not raise.
    events.publish("anything", foo="bar")


def test_publish_calls_registered_publisher_with_envelope():
    captured: list[dict] = []
    events.set_publisher(captured.append)
    try:
        events.publish("dream_complete", dream_id="d1", duration_ms=42)
    finally:
        events.set_publisher(None)
    assert len(captured) == 1
    e = captured[0]
    assert e["type"] == "dream_complete"
    assert e["dream_id"] == "d1"
    assert e["duration_ms"] == 42
    assert "at" in e and e["at"].endswith("Z")  # iso UTC


def test_publish_swallows_publisher_exception(caplog):
    def boom(_event):
        raise RuntimeError("publisher crashed")
    events.set_publisher(boom)
    try:
        events.publish("anything")  # must not raise
    finally:
        events.set_publisher(None)
    assert "event publish failed" in caplog.text


def test_event_bus_publish_is_noop_when_loop_unbound():
    """EventBus.publish before bind_loop should drop silently."""
    bus = events.EventBus()
    q = bus.subscribe()
    bus.publish({"type": "x", "at": "..."})  # loop not bound
    # No event should reach the queue.
    assert q.empty()


@pytest.mark.asyncio
async def test_event_bus_dispatches_to_subscribers():
    bus = events.EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.publish({"type": "ping", "at": "now"})
    # call_soon_threadsafe schedules; one tick yields.
    await asyncio.sleep(0)
    e1 = await asyncio.wait_for(q1.get(), timeout=1)
    e2 = await asyncio.wait_for(q2.get(), timeout=1)
    assert e1 == {"type": "ping", "at": "now"}
    assert e2 == {"type": "ping", "at": "now"}


@pytest.mark.asyncio
async def test_event_bus_drops_oldest_on_overflow(caplog):
    bus = events.EventBus()
    bus.QUEUE_MAX = 2  # shrink for the test
    bus.bind_loop(asyncio.get_running_loop())
    q = bus.subscribe()
    # Don't drain — fill past capacity.
    for i in range(5):
        bus.publish({"type": "n", "i": i, "at": "."})
    await asyncio.sleep(0.01)  # let call_soon_threadsafe land
    received = []
    while not q.empty():
        received.append(q.get_nowait())
    # Oldest dropped: we keep the last 2.
    assert [r["i"] for r in received] == [3, 4]
    assert bus._dropped_total >= 3
```

Note: pytest-asyncio is not in deps. Use the simpler `asyncio` form — wrap async test bodies in `asyncio.run`. Adjust:

```python
def test_event_bus_dispatches_to_subscribers():
    asyncio.run(_dispatch_body())

async def _dispatch_body():
    bus = events.EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.publish({"type": "ping", "at": "now"})
    await asyncio.sleep(0)
    e1 = await asyncio.wait_for(q1.get(), timeout=1)
    e2 = await asyncio.wait_for(q2.get(), timeout=1)
    assert e1 == {"type": "ping", "at": "now"}
    assert e2 == {"type": "ping", "at": "now"}
```

(Same shape for the overflow test.)

Replace the two `@pytest.mark.asyncio` tests with this `asyncio.run`-based form before running.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/bridge/test_events.py -v
```

Expected: `ImportError: cannot import name 'events' from 'brain.bridge'` or `AttributeError: module 'brain.bridge.events' has no attribute 'EventBus'`.

- [ ] **Step 3: Implement `brain/bridge/events.py`**

```python
"""Bridge event bus.

Two layers:

  1. Module-level `set_publisher(fn)` / `publish(type, **payload)` — engines
     call publish; it's a free no-op when no publisher is registered (CLI mode).
  2. EventBus class — bridge instantiates one at startup, calls bind_loop in
     lifespan, then sets it as the module-level publisher. Thread-safe; drops
     oldest event on per-subscriber queue overflow.

OG reference: NellBrain/nell_bridge.py:317-369 (EventBroadcaster).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level publisher contract
# ---------------------------------------------------------------------------

_publisher: Callable[[dict[str, Any]], None] | None = None


def set_publisher(fn: Callable[[dict[str, Any]], None] | None) -> None:
    """Bridge calls this on lifespan startup; sets None on teardown."""
    global _publisher
    _publisher = fn


def publish(event_type: str, **payload: Any) -> None:
    """Engines call this. Free no-op when no publisher is registered.

    Drop semantics: if the publisher raises, the exception is caught and
    logged at WARN. Engines never fail because a publish failed.
    """
    if _publisher is None:
        return
    event = {"type": event_type, "at": _now_iso(), **payload}
    try:
        _publisher(event)
    except Exception:
        logger.warning("event publish failed for type=%s", event_type, exc_info=True)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# EventBus — bridge-side fan-out with thread-safety + drop-on-overflow
# ---------------------------------------------------------------------------


class EventBus:
    """In-process pub/sub for the bridge daemon.

    Subscribers are asyncio.Queues. Publishers may run in any thread (the
    supervisor runs in a non-daemon thread); publish() uses
    call_soon_threadsafe to dispatch onto the bridge's main event loop.

    Per-subscriber queue is bounded at QUEUE_MAX; overflow drops the OLDEST
    event so live clients keep receiving fresh data instead of stale.
    """

    QUEUE_MAX = 64

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped_total = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict) -> None:
        """Dispatch event to every subscriber. Thread-safe.

        Drops silently if loop is not yet bound (shouldn't happen in normal
        bridge lifecycle but is a guard for engine publishes during very
        early startup or very late shutdown).
        """
        if self._loop is None:
            return
        for q in list(self._subscribers):
            self._loop.call_soon_threadsafe(self._enqueue, q, event)

    def _enqueue(self, q: asyncio.Queue, event: dict) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(event)
                self._dropped_total += 1
                if self._dropped_total % 10 == 1:
                    logger.warning(
                        "event queue overflow, dropped=%d", self._dropped_total
                    )
            except asyncio.QueueEmpty:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def subscriber_count(self) -> int:
        return len(self._subscribers)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/bridge/test_events.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Smoke-test the no-op path manually**

```bash
uv run python -c "from brain.bridge.events import publish; publish('test', a=1, b=2); print('no-op ok')"
```

Expected: prints `no-op ok`. (Confirms an engine importing publish from CLI never fails.)

- [ ] **Step 6: Commit**

```bash
git add brain/bridge/events.py tests/bridge/test_events.py
git commit -m "feat(sp7): event bus with thread-safe drop-on-overflow"
```

---

## Task 3: State file module

**Files:**
- Create: `brain/bridge/state_file.py`
- Create: `tests/bridge/test_state_file.py`

The state file is `<persona_dir>/bridge.json`. It's the source of truth for "is the bridge running, on what port, did the last bridge crash." Atomic writes via `save_with_backup`; reads via `attempt_heal`.

- [ ] **Step 1: Write the failing tests**

Create `tests/bridge/test_state_file.py`:

```python
"""Bridge state file — bridge.json schema, atomic writes, recovery predicates."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from brain.bridge import state_file


def test_round_trip_preserves_all_fields(persona_dir: Path):
    state = state_file.BridgeState(
        persona="test-persona",
        pid=os.getpid(),
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, state)
    read_back = state_file.read(persona_dir)
    assert read_back == state


def test_read_returns_none_when_missing(persona_dir: Path):
    assert state_file.read(persona_dir) is None


def test_pid_is_alive_true_for_self():
    assert state_file.pid_is_alive(os.getpid()) is True


def test_pid_is_alive_false_for_dead_pid():
    # A pid that's almost certainly not in use.
    assert state_file.pid_is_alive(999_999) is False


def test_dirty_shutdown_predicate(persona_dir: Path):
    """shutdown_clean: false + dead pid => recovery needed."""
    s = state_file.BridgeState(
        persona="test-persona",
        pid=999_999,  # dead
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.recovery_needed(persona_dir) is True


def test_clean_shutdown_predicate(persona_dir: Path):
    """shutdown_clean: true => recovery NOT needed even if pid was set."""
    s = state_file.BridgeState(
        persona="test-persona",
        pid=None,
        port=None,
        started_at="2026-04-28T10:15:00Z",
        stopped_at="2026-04-28T10:30:00Z",
        shutdown_clean=True,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.recovery_needed(persona_dir) is False


def test_running_predicate_with_live_pid(persona_dir: Path):
    s = state_file.BridgeState(
        persona="test-persona",
        pid=os.getpid(),
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.is_running(persona_dir) is True


def test_running_predicate_with_dead_pid(persona_dir: Path):
    s = state_file.BridgeState(
        persona="test-persona",
        pid=999_999,
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.is_running(persona_dir) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/bridge/test_state_file.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.bridge.state_file'`.

- [ ] **Step 3: Implement `brain/bridge/state_file.py`**

```python
"""Bridge daemon state file — bridge.json.

The file is persistent across restarts. `pid`/`port` get cleared on graceful
shutdown; `shutdown_clean` flips to True as the very last step. On the next
`bridge start`, if `shutdown_clean: false` AND the recorded pid is dead, the
previous bridge crashed — caller should run dirty-shutdown recovery.

Atomic writes go through brain.health.attempt_heal.save_with_backup so
partial writes don't corrupt state. Reads use attempt_heal so corrupted
files restore from .bak1.

OG reference: NellBrain/nell_supervisor.py:53-101 (state file machinery,
shutdown_clean flag, pid helpers).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from brain.health.adaptive import compute_treatment
from brain.health.attempt_heal import attempt_heal, save_with_backup

logger = logging.getLogger(__name__)

STATE_FILENAME = "bridge.json"


@dataclass
class BridgeState:
    """Full bridge.json schema."""

    persona: str
    pid: int | None
    port: int | None
    started_at: str
    stopped_at: str | None
    shutdown_clean: bool
    client_origin: str  # "cli" | "tauri" | "tests"


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / STATE_FILENAME


def _default_factory() -> dict:
    """Returned when bridge.json is missing AND all .bak files are corrupt.

    A "missing" bridge.json (never written) returns None from read() — we
    only fall through to default_factory in attempt_heal when the file
    exists but is unrecoverable. In that case we return a stub that
    recovery_needed() sees as 'not running, not dirty'.
    """
    return {
        "persona": "",
        "pid": None,
        "port": None,
        "started_at": "",
        "stopped_at": None,
        "shutdown_clean": True,
        "client_origin": "cli",
    }


def write(persona_dir: Path, state: BridgeState) -> None:
    """Atomically write bridge.json with .bak rotation."""
    path = _state_path(persona_dir)
    treatment = compute_treatment(persona_dir, STATE_FILENAME)
    save_with_backup(path, asdict(state), backup_count=treatment.backup_count)


def read(persona_dir: Path) -> BridgeState | None:
    """Read bridge.json. Returns None if file does not exist.

    If the file exists but is corrupt, attempt_heal restores from .bak rotation
    and returns the recovered data. If all backups are corrupt, returns the
    default-factory output (treated as 'not running, clean shutdown').
    """
    path = _state_path(persona_dir)
    if not path.exists():
        return None
    data, anomaly = attempt_heal(path, _default_factory)
    if anomaly is not None:
        logger.warning(
            "bridge.json anomaly: %s (%s) — using recovered/default state",
            anomaly.kind,
            anomaly.action,
        )
    return BridgeState(**data)


def pid_is_alive(pid: int) -> bool:
    """Return True if the given pid is a live process owned by this user."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Pid exists but is owned by someone else — treat as alive.
        return True


def recovery_needed(persona_dir: Path) -> bool:
    """True iff the previous bridge process exited dirty.

    Predicate: state file exists AND shutdown_clean is False AND
    the recorded pid is dead.
    """
    state = read(persona_dir)
    if state is None:
        return False
    if state.shutdown_clean:
        return False
    if state.pid is None:
        return False
    return not pid_is_alive(state.pid)


def is_running(persona_dir: Path) -> bool:
    """True iff a live bridge daemon is recorded for this persona."""
    state = read(persona_dir)
    if state is None or state.pid is None:
        return False
    return pid_is_alive(state.pid)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/bridge/test_state_file.py -v
```

Expected: `8 passed`.

- [ ] **Step 5: Smoke-test the round-trip manually**

```bash
uv run python -c "
from pathlib import Path
import tempfile, os
from brain.bridge import state_file
with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    s = state_file.BridgeState(persona='x', pid=os.getpid(), port=51234, started_at='2026-04-28T00:00:00Z', stopped_at=None, shutdown_clean=False, client_origin='cli')
    state_file.write(p, s)
    print('written:', (p/'bridge.json').read_text())
    print('is_running:', state_file.is_running(p))
"
```

Expected: prints the JSON round-trip, then `is_running: True`.

- [ ] **Step 6: Commit**

```bash
git add brain/bridge/state_file.py tests/bridge/test_state_file.py
git commit -m "feat(sp7): bridge.json state file with dirty-shutdown predicates"
```

---

## Task 4: FastAPI server skeleton — health, session/new, state

**Files:**
- Create: `brain/bridge/server.py`
- Create: `tests/bridge/test_endpoints.py` (lifecycle endpoints only, chat in Task 5)

This task scaffolds the FastAPI app: lifespan (binds EventBus loop, sets module publisher, builds singletons), three lifecycle endpoints, and global state held on `app.state`. Chat endpoints come in Task 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/bridge/test_endpoints.py`:

```python
"""Bridge endpoints — sync TestClient against an in-memory FastAPI app."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _make_client(persona_dir: Path) -> TestClient:
    """Build the FastAPI app pinned to a tmp persona and return a TestClient."""
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return TestClient(app)


def test_health_returns_ok(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["liveness"] == "ok"
        assert body["persona"] == persona_dir.name
        assert "uptime_s" in body
        assert body["sessions_active"] == 0
        assert "pending_alarms" in body  # from walk_persona/compute_pending_alarms
        assert body["supervisor_thread"] in ("not-started", "alive", "dead")


def test_session_new_returns_uuid_and_tracks_state(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.post("/session/new", json={"client": "tests"})
        assert r.status_code == 200
        body = r.json()
        sid = body["session_id"]
        assert len(sid) == 36  # uuid4 string
        assert body["persona"] == persona_dir.name
        assert "created_at" in body

        # /state/{sid} should now find it
        r2 = c.get(f"/state/{sid}")
        assert r2.status_code == 200
        s = r2.json()
        assert s["session_id"] == sid
        assert s["turns"] == 0
        assert s["history_len"] == 0
        assert s["in_flight"] is False


def test_state_returns_404_for_unknown_session(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.get("/state/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404


def test_health_sessions_active_increments_after_new(persona_dir: Path):
    with _make_client(persona_dir) as c:
        c.post("/session/new", json={"client": "tests"})
        c.post("/session/new", json={"client": "tests"})
        r = c.get("/health")
        assert r.json()["sessions_active"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/bridge/test_endpoints.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.bridge.server'`.

- [ ] **Step 3: Implement `brain/bridge/server.py`**

```python
"""SP-7 FastAPI app — bridge daemon HTTP+WS server.

Exposes:
  POST /session/new        — create a new session
  GET  /state/{session_id} — return session state
  GET  /health             — liveness + walk_persona + alarms

Chat endpoints (POST /chat, WS /stream, WS /events, POST /sessions/close) are
added in Task 5+. This task scaffolds the app and the lifecycle endpoints only.

Singletons are constructed once at lifespan startup and held on app.state:
  - MemoryStore, HebbianMatrix, EmbeddingCache, LLMProvider
  - EventBus
  - SessionRegistry (in-memory dict[session_id, SessionState])
  - in_flight_locks: dict[session_id, asyncio.Lock]
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from brain.bridge import events
from brain.bridge.events import EventBus
from brain.bridge.provider import LLMProvider, build_provider
from brain.chat.session import SessionState, create_session, get_session
from brain.health.alarm import compute_pending_alarms
from brain.health.walker import walk_persona
from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class NewSessionReq(BaseModel):
    client: str = "cli"  # "cli" | "tauri" | "tests"


class NewSessionResp(BaseModel):
    session_id: str
    persona: str
    created_at: str


# ---------------------------------------------------------------------------
# App state container — held on app.state
# ---------------------------------------------------------------------------


@dataclass
class BridgeAppState:
    persona_dir: Path
    persona: str
    client_origin: str
    started_at: datetime
    store: MemoryStore
    hebbian: HebbianMatrix
    embeddings: EmbeddingCache
    provider: LLMProvider
    event_bus: EventBus
    in_flight_locks: dict[str, asyncio.Lock]
    last_chat_at: datetime | None = None
    supervisor_thread: Any | None = None  # filled by Task 6


# ---------------------------------------------------------------------------
# Lifespan + app factory
# ---------------------------------------------------------------------------


def build_app(persona_dir: Path, client_origin: str = "cli") -> FastAPI:
    """Build a FastAPI app for the given persona. Public for tests + daemon."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Construct singletons.
        store = MemoryStore(persona_dir / "memories.db")
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        embeddings = EmbeddingCache()
        provider = build_provider(persona_dir)

        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())
        events.set_publisher(bus.publish)

        app.state.bridge = BridgeAppState(
            persona_dir=persona_dir,
            persona=persona_dir.name,
            client_origin=client_origin,
            started_at=datetime.now(UTC),
            store=store,
            hebbian=hebbian,
            embeddings=embeddings,
            provider=provider,
            event_bus=bus,
            in_flight_locks={},
        )
        logger.info("bridge started persona=%s pid=%d", persona_dir.name, os.getpid())
        try:
            yield
        finally:
            events.set_publisher(None)
            store.close()
            hebbian.close()
            logger.info("bridge stopped persona=%s", persona_dir.name)

    app = FastAPI(title="companion-emergence bridge", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        uptime = (datetime.now(UTC) - s.started_at).total_seconds()

        # Walk + alarms — lightweight
        try:
            anomalies = walk_persona(s.persona_dir)
            alarms = compute_pending_alarms(s.persona_dir)
        except Exception:
            logger.warning("health walk failed", exc_info=True)
            anomalies = []
            alarms = []

        sup_thread = s.supervisor_thread
        if sup_thread is None:
            sup_status = "not-started"
        elif sup_thread.is_alive():
            sup_status = "alive"
        else:
            sup_status = "dead"

        return {
            "liveness": "ok",
            "persona": s.persona,
            "uptime_s": int(uptime),
            "pid": os.getpid(),
            "sessions_active": _session_count(),
            "last_chat_at": s.last_chat_at.isoformat() if s.last_chat_at else None,
            "supervisor_thread": sup_status,
            "pending_alarms": len(alarms),
            "anomalies": len(anomalies),
            "shutdown_clean_last": True,  # Task 7 will surface previous-state info
        }

    @app.post("/session/new", response_model=NewSessionResp)
    def session_new(req: NewSessionReq) -> NewSessionResp:
        s: BridgeAppState = app.state.bridge
        sess = create_session(s.persona)
        return NewSessionResp(
            session_id=sess.session_id,
            persona=s.persona,
            created_at=sess.created_at.isoformat(),
        )

    @app.get("/state/{session_id}")
    def state_endpoint(session_id: str) -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        sess = get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="session not found")
        in_flight = session_id in s.in_flight_locks and s.in_flight_locks[session_id].locked()
        return {
            "session_id": sess.session_id,
            "persona": sess.persona_name,
            "turns": sess.turns,
            "last_turn_at": sess.last_turn_at.isoformat() if sess.last_turn_at else None,
            "history_len": len(sess.history),
            "in_flight": in_flight,
        }

    return app


def _session_count() -> int:
    """Count of registered sessions across the in-memory registry."""
    from brain.chat.session import all_sessions

    return len(all_sessions())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/bridge/test_endpoints.py -v
```

Expected: `4 passed`. If `build_provider` fails because no provider config, your test persona dir is empty — add a `persona_config.json` fixture in conftest. (See if the test fails on that and react; otherwise move on.)

- [ ] **Step 5: Smoke-test by booting the app and curling**

Run the server in one terminal:

```bash
NELLBRAIN_HOME=/tmp/sp7-smoke uv run python -c "
import os
from pathlib import Path
from brain.bridge.server import build_app
import uvicorn

p = Path('/tmp/sp7-smoke/personas/smoketest')
p.mkdir(parents=True, exist_ok=True)
(p / 'active_conversations').mkdir(exist_ok=True)
app = build_app(persona_dir=p, client_origin='cli')
uvicorn.run(app, host='127.0.0.1', port=8765, log_level='warning')
" &
sleep 2
```

Then in another terminal:

```bash
curl -s http://127.0.0.1:8765/health | python -m json.tool
curl -s -X POST http://127.0.0.1:8765/session/new -H 'content-type: application/json' -d '{"client":"smoke"}' | python -m json.tool
```

Expected:
- `/health` returns JSON with `"liveness": "ok"`, `"persona": "smoketest"`
- `/session/new` returns a UUID

Kill the server: `pkill -f 'brain.bridge.server'` (or whichever pattern matches).

- [ ] **Step 6: Commit**

```bash
git add brain/bridge/server.py tests/bridge/test_endpoints.py
git commit -m "feat(sp7): FastAPI app skeleton — health/session/state endpoints"
```

---

## Task 5: Chat endpoints — POST /chat + WS /stream + WS /events + POST /sessions/close

**Files:**
- Modify: `brain/bridge/server.py` — add four endpoint handlers
- Create: `tests/bridge/test_event_stream.py` — `/events` WS tests
- Modify: `tests/bridge/test_endpoints.py` — add chat endpoint tests

The chat path: `POST /chat` is a JSON one-shot fallback. `WS /stream/{sid}` simulates streaming by chunking the full response word-by-word. Tool events fire BEFORE reply chunks (they actually happened first inside the tool loop). `POST /sessions/close` triggers ingest. `WS /events` is server-push only with a `connected` greeting.

- [ ] **Step 1: Write the failing tests for /chat + /sessions/close**

Append to `tests/bridge/test_endpoints.py`:

```python
def test_chat_404_unknown_session(persona_dir: Path):
    with _make_client(persona_dir) as c:
        r = c.post("/chat", json={"session_id": "no-such-sid", "message": "hi"})
        assert r.status_code == 404


def test_chat_round_trip_with_fake_provider(persona_dir: Path, monkeypatch):
    """Mock provider returns a fixed reply; verify /chat returns it and history persists."""
    _patch_fake_provider(monkeypatch, reply="hello, hana")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        r = c.post("/chat", json={"session_id": sid, "message": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["reply"] == "hello, hana"
        assert body["turn"] == 1
        assert "duration_ms" in body

        # State reflects the turn
        s = c.get(f"/state/{sid}").json()
        assert s["turns"] == 1
        assert s["history_len"] == 2  # user + assistant


def test_chat_429_when_session_in_flight(persona_dir: Path, monkeypatch):
    """If we hold the per-session lock, /chat should refuse with 429."""
    _patch_fake_provider(monkeypatch, reply="...")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        # Manually acquire the lock to simulate in-flight.
        bridge_state = c.app.state.bridge
        lock = bridge_state.in_flight_locks.setdefault(sid, _new_lock_in_loop())
        # Mark locked by acquiring without releasing — done in a helper.
        _force_lock_acquire(lock)

        r = c.post("/chat", json={"session_id": sid, "message": "hi"})
        assert r.status_code == 429


def test_sessions_close_returns_ingest_report(persona_dir: Path, monkeypatch):
    """Close after a chat: should return committed/deduped/etc counts."""
    _patch_fake_provider(monkeypatch, reply="goodbye")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        c.post("/chat", json={"session_id": sid, "message": "see you"})
        r = c.post("/sessions/close", json={"session_id": sid})
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        # With FakeProvider extracting nothing meaningful, committed may be 0;
        # the shape is what we care about.
        assert "committed" in body
        assert "deduped" in body
        assert "errors" in body
```

Add a `_patch_fake_provider` helper at the top of the test file:

```python
import asyncio

from brain.bridge.chat import ChatResponse


def _patch_fake_provider(monkeypatch, reply: str):
    """Replace LLMProvider with a stub that returns `reply` and no tools."""
    import brain.bridge.server as srv

    class _Fake:
        def name(self) -> str:
            return "fake"

        def chat(self, messages, *, tools=None, options=None):
            return ChatResponse(content=reply, tool_calls=[])

        def generate(self, prompt, *, system=None):
            return reply

    monkeypatch.setattr(srv, "build_provider", lambda _p: _Fake())


def _new_lock_in_loop() -> asyncio.Lock:
    """Construct an asyncio.Lock outside any running loop. Caller may set as
    in_flight_locks[sid] before sending a request."""
    return asyncio.Lock()


def _force_lock_acquire(lock: asyncio.Lock) -> None:
    """Synchronously mark an asyncio.Lock as held (for the 429 test)."""

    async def _acq():
        await lock.acquire()

    # Use the TestClient's event loop indirectly via run_until_complete on a
    # fresh loop is fine here because TestClient holds its own.
    asyncio.new_event_loop().run_until_complete(_acq())
```

(If lock-coupling proves flaky in practice, replace this test with a 2-call concurrent `httpx.AsyncClient` test; keep the design and the assertion the same.)

- [ ] **Step 2: Write the failing tests for /stream**

Add to `tests/bridge/test_endpoints.py`:

```python
def test_stream_round_trip(persona_dir: Path, monkeypatch):
    """WS /stream/{sid} sends started, reply_chunks, done."""
    _patch_fake_provider(monkeypatch, reply="hello world from nell")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "hi"})
            frames = []
            while True:
                f = ws.receive_json()
                frames.append(f)
                if f.get("type") == "done":
                    break

        types = [f["type"] for f in frames]
        assert types[0] == "started"
        assert types[-1] == "done"
        assert "reply_chunk" in types
        # Reassemble chunks
        chunked = "".join(
            f["text"] for f in frames if f.get("type") == "reply_chunk"
        )
        assert chunked == "hello world from nell"


def test_stream_session_busy_when_locked(persona_dir: Path, monkeypatch):
    _patch_fake_provider(monkeypatch, reply="x")
    with _make_client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        lock = c.app.state.bridge.in_flight_locks.setdefault(sid, _new_lock_in_loop())
        _force_lock_acquire(lock)
        with c.websocket_connect(f"/stream/{sid}") as ws:
            ws.send_json({"message": "hi"})
            f = ws.receive_json()
            assert f["type"] == "error"
            assert f["code"] == "session_busy"
            assert f["done"] is True
```

- [ ] **Step 3: Write the failing tests for /events**

Create `tests/bridge/test_event_stream.py`:

```python
"""Bridge /events WS — server-pushed broadcast tests."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _client(persona_dir: Path) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests"))


def test_events_greets_with_connected(persona_dir: Path):
    with _client(persona_dir) as c:
        with c.websocket_connect("/events") as ws:
            f = ws.receive_json()
            assert f["type"] == "connected"
            assert f["subscribers"] >= 1


def test_events_receives_chat_done_after_chat(persona_dir: Path, monkeypatch):
    """A chat turn must emit chat_done on /events."""
    from tests.bridge.test_endpoints import _patch_fake_provider

    _patch_fake_provider(monkeypatch, reply="hi back")
    with _client(persona_dir) as c:
        with c.websocket_connect("/events") as evt_ws:
            # Drain the connected greeting
            evt_ws.receive_json()
            sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
            c.post("/chat", json={"session_id": sid, "message": "hello"})
            # Drain until we see chat_done (skip other event types)
            seen_types = set()
            for _ in range(20):
                f = evt_ws.receive_json()
                seen_types.add(f["type"])
                if f["type"] == "chat_done":
                    return
            raise AssertionError(f"chat_done never seen; saw: {seen_types}")
```

- [ ] **Step 4: Run all bridge tests; confirm new ones fail with the right error**

```bash
uv run pytest tests/bridge/ -v
```

Expected: existing 12 pass; the new ~7 fail with `AttributeError` on missing routes / `404 Not Found` for unimplemented endpoints.

- [ ] **Step 5: Implement the four endpoints in `brain/bridge/server.py`**

Append inside `build_app(...)`, after the existing `state_endpoint`:

```python
    # ── POST /chat — JSON one-shot fallback ────────────────────────────
    class ChatReq(BaseModel):
        session_id: str
        message: str

    @app.post("/chat")
    async def chat(req: ChatReq) -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        sess = get_session(req.session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="session not found")
        lock = s.in_flight_locks.setdefault(req.session_id, asyncio.Lock())
        if lock.locked():
            raise HTTPException(status_code=429, detail="session has an in-flight turn")
        async with lock:
            t0 = datetime.now(UTC)
            events.publish("chat_started", session_id=req.session_id, client=s.client_origin)
            try:
                result = await asyncio.to_thread(
                    _respond_blocking, s, sess, req.message
                )
            except Exception as exc:
                logger.exception("chat failed session=%s", req.session_id)
                raise HTTPException(status_code=502, detail=f"provider error: {exc}") from exc
            duration_ms = int((datetime.now(UTC) - t0).total_seconds() * 1000)
            s.last_chat_at = datetime.now(UTC)
            events.publish(
                "chat_done",
                session_id=req.session_id,
                turn=result.turn,
                duration_ms=duration_ms,
            )
            return {
                "session_id": req.session_id,
                "reply": result.content,
                "turn": result.turn,
                "tool_invocations": result.tool_invocations,
                "duration_ms": duration_ms,
            }

    # ── WS /stream/{session_id} — simulated streaming ───────────────────
    from fastapi import WebSocket, WebSocketDisconnect

    chunk_delay_ms = int(os.environ.get("NELL_STREAM_CHUNK_DELAY_MS", "30"))

    @app.websocket("/stream/{session_id}")
    async def stream(ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        s: BridgeAppState = app.state.bridge
        sess = get_session(session_id)
        if sess is None:
            await ws.send_json({"type": "error", "code": "session_not_found", "done": True})
            await ws.close()
            return
        lock = s.in_flight_locks.setdefault(session_id, asyncio.Lock())
        if lock.locked():
            await ws.send_json({"type": "error", "code": "session_busy", "done": True})
            await ws.close()
            return

        try:
            req = await ws.receive_json()
        except (WebSocketDisconnect, ValueError):
            return
        message = req.get("message", "")
        if not message:
            await ws.send_json({"type": "error", "code": "empty_message", "done": True})
            await ws.close()
            return

        async with lock:
            t0 = datetime.now(UTC)
            await ws.send_json({"type": "started", "session_id": session_id, "at": _now()})
            events.publish("chat_started", session_id=session_id, client=s.client_origin)

            try:
                result = await asyncio.to_thread(_respond_blocking, s, sess, message)
            except Exception as exc:
                logger.exception("stream failed session=%s", session_id)
                await ws.send_json(
                    {"type": "error", "code": "provider_failed", "detail": str(exc), "done": True}
                )
                await ws.close()
                return

            # Tool events first — they happened first inside the tool loop.
            for inv in result.tool_invocations:
                await ws.send_json(
                    {"type": "tool_call", "tool": inv.get("name", "?"), "session_id": session_id, "at": _now()}
                )
                await ws.send_json(
                    {
                        "type": "tool_result",
                        "tool": inv.get("name", "?"),
                        "summary": inv.get("summary", ""),
                        "session_id": session_id,
                        "at": _now(),
                    }
                )

            # Word-by-word chunking
            for word in _word_chunks(result.content):
                await ws.send_json({"type": "reply_chunk", "text": word})
                if chunk_delay_ms > 0:
                    await asyncio.sleep(chunk_delay_ms / 1000.0)

            duration_ms = int((datetime.now(UTC) - t0).total_seconds() * 1000)
            s.last_chat_at = datetime.now(UTC)
            await ws.send_json(
                {
                    "type": "done",
                    "session_id": session_id,
                    "turn": result.turn,
                    "duration_ms": duration_ms,
                    "at": _now(),
                }
            )
            events.publish(
                "chat_done",
                session_id=session_id,
                turn=result.turn,
                duration_ms=duration_ms,
            )

    # ── WS /events — server-push only ─────────────────────────────────
    @app.websocket("/events")
    async def events_ws(ws: WebSocket) -> None:
        await ws.accept()
        s: BridgeAppState = app.state.bridge
        q = s.event_bus.subscribe()
        await ws.send_json(
            {"type": "connected", "subscribers": s.event_bus.subscriber_count(), "at": _now()}
        )
        try:
            while True:
                event = await q.get()
                await ws.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            s.event_bus.unsubscribe(q)

    # ── POST /sessions/close — explicit ingest trigger ────────────────
    class CloseReq(BaseModel):
        session_id: str

    @app.post("/sessions/close")
    async def sessions_close(req: CloseReq) -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        from brain.ingest.pipeline import close_session

        report = await asyncio.to_thread(
            close_session,
            s.persona_dir,
            req.session_id,
            store=s.store,
            hebbian=s.hebbian,
            provider=s.provider,
            embeddings=s.embeddings,
        )
        events.publish(
            "session_closed",
            session_id=req.session_id,
            committed=report.committed,
            deduped=report.deduped,
            soul_candidates=report.soul_candidates,
            errors=report.errors,
        )
        return {
            "session_id": req.session_id,
            "committed": report.committed,
            "deduped": report.deduped,
            "soul_candidates": report.soul_candidates,
            "errors": report.errors,
        }

    return app
```

Add the helper functions at module level (after `_session_count`):

```python
def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _word_chunks(text: str) -> list[str]:
    """Split into word-or-whitespace tokens preserving spacing.

    'hello world  from' -> ['hello', ' ', 'world', '  ', 'from']
    Each chunk is sent verbatim so reassembly == original text.
    """
    import re

    return re.findall(r"\S+|\s+", text)


def _respond_blocking(s: BridgeAppState, sess: SessionState, message: str):
    """Wrap brain.chat.engine.respond — blocks; called via to_thread."""
    from brain.chat.engine import respond

    return respond(
        s.persona_dir,
        message,
        store=s.store,
        hebbian=s.hebbian,
        provider=s.provider,
        session=sess,
    )
```

- [ ] **Step 6: Run all tests**

```bash
uv run pytest tests/bridge/ -v
```

Expected: previously 12 + new ~7 = ~19 pass.

If `_force_lock_acquire` is flaky (locks have loop affinity), replace those two tests with `httpx.AsyncClient` concurrent-call form. Keep the design intact.

- [ ] **Step 7: Smoke-test streaming over a real WS**

Run the server (same one-liner as Task 4 step 5). In another terminal:

```bash
uv run python -c "
from websockets.sync.client import connect
import json
ws = connect('ws://127.0.0.1:8765/stream/replace-with-real-sid')
ws.send(json.dumps({'message': 'hi'}))
while True:
    msg = json.loads(ws.recv())
    print(msg)
    if msg.get('type') == 'done':
        break
"
```

(Get a real sid first via `curl -X POST http://127.0.0.1:8765/session/new -H 'content-type: application/json' -d '{}'`.) Expected: see `started` → `reply_chunk`s → `done`.

- [ ] **Step 8: Commit**

```bash
git add brain/bridge/server.py tests/bridge/test_endpoints.py tests/bridge/test_event_stream.py
git commit -m "feat(sp7): chat endpoints — /chat, /stream simulated, /events, /sessions/close"
```

---

## Task 6: Supervisor thread + idle-shutdown watcher

**Files:**
- Create: `brain/bridge/supervisor.py`
- Modify: `brain/bridge/server.py` — start supervisor in lifespan; add idle-shutdown watcher
- Create: `tests/bridge/test_lifecycle.py`

The supervisor is a non-daemon thread running a 60s tick (configurable to 0.5s for tests). Each tick calls `close_stale_sessions`, publishes `supervisor_tick` and `session_closed` events. The idle-shutdown watcher is an async task in the main loop; checks `last_chat_at` and triggers graceful shutdown.

- [ ] **Step 1: Write failing tests for the supervisor + idle**

Create `tests/bridge/test_lifecycle.py`:

```python
"""Bridge lifecycle — supervisor tick, idle-shutdown, graceful close."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app


def _client(persona_dir: Path, **kw) -> TestClient:
    return TestClient(build_app(persona_dir=persona_dir, client_origin="tests", **kw))


def test_supervisor_tick_emits_event(persona_dir: Path):
    """With tick_interval_s=0.2, /events sees supervisor_tick within ~1s."""
    with _client(persona_dir, tick_interval_s=0.2) as c:
        with c.websocket_connect("/events") as ws:
            ws.receive_json()  # connected greeting
            seen = False
            t0 = time.time()
            while time.time() - t0 < 3:
                f = ws.receive_json()
                if f["type"] == "supervisor_tick":
                    seen = True
                    break
            assert seen, "supervisor_tick not received within 3s"


def test_graceful_shutdown_closes_active_sessions(persona_dir: Path, monkeypatch):
    """With one chat session in flight, lifespan teardown drains it via close_stale_sessions(silence_minutes=0)."""
    from tests.bridge.test_endpoints import _patch_fake_provider

    _patch_fake_provider(monkeypatch, reply="bye")
    closed = []
    from brain.ingest import pipeline as ingest_pipeline

    real_close = ingest_pipeline.close_stale_sessions

    def spy(persona_dir, *, silence_minutes, **kw):
        closed.append(silence_minutes)
        return real_close(persona_dir, silence_minutes=silence_minutes, **kw)

    monkeypatch.setattr(ingest_pipeline, "close_stale_sessions", spy)

    with _client(persona_dir) as c:
        sid = c.post("/session/new", json={"client": "tests"}).json()["session_id"]
        c.post("/chat", json={"session_id": sid, "message": "hi"})
    # __exit__ already ran — assert close_stale_sessions(silence_minutes=0) was called
    assert 0 in closed or 0.0 in closed


def test_idle_shutdown_fires_after_no_traffic(persona_dir: Path):
    """idle_shutdown_seconds=1 + no chat → bridge exits soon after."""
    # This test is best-effort given TestClient doesn't run an external loop;
    # verify the watcher predicate fires by calling _check_idle directly.
    from brain.bridge.server import _check_idle

    class FakeState:
        last_chat_at = None
        in_flight_locks: dict = {}

    s = FakeState()
    assert _check_idle(s, idle_shutdown_seconds=1) is True  # never chatted

    from datetime import UTC, datetime, timedelta

    s.last_chat_at = datetime.now(UTC) - timedelta(seconds=5)
    assert _check_idle(s, idle_shutdown_seconds=1) is True

    s.last_chat_at = datetime.now(UTC)
    assert _check_idle(s, idle_shutdown_seconds=1) is False  # too fresh
```

- [ ] **Step 2: Run; confirm failures**

```bash
uv run pytest tests/bridge/test_lifecycle.py -v
```

Expected: errors importing things that don't yet exist (`tick_interval_s` kwarg, `_check_idle`).

- [ ] **Step 3: Implement `brain/bridge/supervisor.py`**

```python
"""SP-7 supervisor thread — folded as non-daemon thread inside the bridge.

run_folded() is a synchronous loop: every `tick_interval_s` it calls
close_stale_sessions and publishes events. Lives in a separate thread so the
async server stays responsive; uses event_bus.publish (thread-safe) to fan
out events to /events subscribers.

Non-daemon thread on purpose — SIGTERM must wait for the loop to exit
before process exit, so we don't kill mid-ingest.

OG reference: NellBrain/nell_supervisor.py:368-407 (run_folded).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from brain.bridge.events import EventBus
from brain.bridge.provider import LLMProvider
from brain.ingest.pipeline import close_stale_sessions
from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def run_folded(
    stop_event: threading.Event,
    *,
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache,
    event_bus: EventBus,
    tick_interval_s: float = 60.0,
    silence_minutes: float = 5.0,
) -> None:
    """Run close_stale_sessions every tick_interval_s until stop_event is set."""
    logger.info("supervisor folded persona=%s tick=%.2fs", persona_dir.name, tick_interval_s)
    while not stop_event.is_set():
        try:
            reports = close_stale_sessions(
                persona_dir,
                silence_minutes=silence_minutes,
                store=store,
                hebbian=hebbian,
                provider=provider,
                embeddings=embeddings,
            )
            for r in reports:
                event_bus.publish(
                    {
                        "type": "session_closed",
                        "session_id": r.session_id,
                        "committed": r.committed,
                        "deduped": r.deduped,
                        "soul_candidates": r.soul_candidates,
                        "errors": r.errors,
                        "at": _now_iso(),
                    }
                )
            event_bus.publish(
                {
                    "type": "supervisor_tick",
                    "closed_sessions": len(reports),
                    "next_tick_in_s": tick_interval_s,
                    "at": _now_iso(),
                }
            )
        except Exception:
            logger.exception("supervisor tick raised")
        # Sleep in 0.1s slices so stop_event is responsive.
        slept = 0.0
        while slept < tick_interval_s and not stop_event.is_set():
            time.sleep(0.1)
            slept += 0.1
    logger.info("supervisor stopped persona=%s", persona_dir.name)


def _now_iso() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
```

- [ ] **Step 4: Wire supervisor + idle-shutdown into `build_app`**

Modify `brain/bridge/server.py` `build_app` signature and lifespan:

```python
def build_app(
    persona_dir: Path,
    client_origin: str = "cli",
    tick_interval_s: float = 60.0,
    silence_minutes: float = 5.0,
    idle_shutdown_seconds: float | None = None,
) -> FastAPI:
    """...existing docstring..."""
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # ... existing singleton construction ...

        # Spawn supervisor thread
        from brain.bridge.supervisor import run_folded

        stop_event = threading.Event()
        sup_thread = threading.Thread(
            target=run_folded,
            kwargs={
                "stop_event": stop_event,
                "persona_dir": persona_dir,
                "store": store,
                "hebbian": hebbian,
                "provider": provider,
                "embeddings": embeddings,
                "event_bus": bus,
                "tick_interval_s": tick_interval_s,
                "silence_minutes": silence_minutes,
            },
            name="sp7-supervisor",
            daemon=False,
        )
        sup_thread.start()
        app.state.bridge.supervisor_thread = sup_thread
        app.state.bridge._supervisor_stop = stop_event

        # Idle-shutdown watcher (only if requested)
        idle_task = None
        if idle_shutdown_seconds is not None and idle_shutdown_seconds > 0:
            idle_task = asyncio.create_task(
                _idle_watcher(app.state.bridge, idle_shutdown_seconds)
            )

        try:
            yield
        finally:
            if idle_task is not None:
                idle_task.cancel()
            stop_event.set()
            sup_thread.join(timeout=180.0)
            if sup_thread.is_alive():
                logger.warning("supervisor thread did not stop within 180s")
            # Drain all live sessions before shutdown
            try:
                from brain.ingest.pipeline import close_stale_sessions

                reports = close_stale_sessions(
                    persona_dir,
                    silence_minutes=0,
                    store=store,
                    hebbian=hebbian,
                    provider=provider,
                    embeddings=embeddings,
                )
                bus.publish(
                    {"type": "shutdown", "clean": True, "drained": len(reports), "at": _now()}
                )
            except Exception:
                logger.exception("shutdown drain failed")
            events.set_publisher(None)
            store.close()
            hebbian.close()
            logger.info("bridge stopped persona=%s", persona_dir.name)

    # ... rest of build_app unchanged ...
```

Also add `import threading` at the top.

Add the helper function at module level:

```python
async def _idle_watcher(state: BridgeAppState, idle_shutdown_seconds: float) -> None:
    """Background task that triggers graceful shutdown after idle threshold."""
    while True:
        await asyncio.sleep(min(idle_shutdown_seconds, 60))
        if _check_idle(state, idle_shutdown_seconds):
            logger.info("idle shutdown firing — no traffic for >%ss", idle_shutdown_seconds)
            os.kill(os.getpid(), 15)  # SIGTERM, lifespan will run cleanup
            return


def _check_idle(state: BridgeAppState, idle_shutdown_seconds: float) -> bool:
    """True if bridge should auto-shutdown.

    Conditions:
      - last_chat_at is None OR older than threshold
      - no active session has its in_flight lock held
    """
    now = datetime.now(UTC)
    if state.last_chat_at is not None:
        if (now - state.last_chat_at).total_seconds() < idle_shutdown_seconds:
            return False
    for lock in state.in_flight_locks.values():
        if lock.locked():
            return False
    return True
```

- [ ] **Step 5: Run all bridge tests**

```bash
uv run pytest tests/bridge/ -v
```

Expected: ~22 pass.

- [ ] **Step 6: Smoke-test the supervisor with a short tick**

```bash
NELLBRAIN_HOME=/tmp/sp7-smoke uv run python -c "
from pathlib import Path
from brain.bridge.server import build_app
import uvicorn

p = Path('/tmp/sp7-smoke/personas/smoketest')
p.mkdir(parents=True, exist_ok=True)
(p / 'active_conversations').mkdir(exist_ok=True)
app = build_app(persona_dir=p, tick_interval_s=2, silence_minutes=0.5)
uvicorn.run(app, host='127.0.0.1', port=8765, log_level='info')
" &
sleep 3
```

Then watch the events:

```bash
uv run python -c "
from websockets.sync.client import connect
import json, time
ws = connect('ws://127.0.0.1:8765/events')
t0 = time.time()
while time.time() - t0 < 8:
    print(json.loads(ws.recv()))
"
```

Expected: see `connected` then several `supervisor_tick` events at ~2s intervals.

Kill the server.

- [ ] **Step 7: Commit**

```bash
git add brain/bridge/supervisor.py brain/bridge/server.py tests/bridge/test_lifecycle.py
git commit -m "feat(sp7): supervisor thread + idle-shutdown watcher + graceful drain"
```

---

## Task 7: CLI integration + dirty-shutdown recovery + auto-spawn

**Files:**
- Create: `brain/bridge/daemon.py` — `cmd_start`, `cmd_stop`, `cmd_status`, `cmd_tail`, recovery
- Modify: `brain/cli.py` — add `nell bridge` subcommand group; add `--no-bridge`/`--bridge-only` to `nell chat` and route via bridge by default

This task lights up the user-facing surface and stitches dirty-shutdown recovery into startup. The recovery path: on `bridge start`, if `state_file.recovery_needed(persona_dir)`, run `close_stale_sessions(silence_minutes=0)` BEFORE spawning the new server.

- [ ] **Step 1: Write the failing tests for daemon orchestration**

Append to `tests/bridge/test_lifecycle.py`:

```python
def test_dirty_shutdown_drains_orphan_buffers(persona_dir: Path, monkeypatch):
    """Pre-write a stale bridge.json + a session buffer; recovery should drain it."""
    import os

    from brain.bridge import daemon, state_file
    from brain.ingest.buffer import ingest_turn

    # Plant a stale state file: dead pid + shutdown_clean=False
    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=999_999,
        port=51234,
        started_at="2026-04-28T10:00:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)

    # Plant a conversation buffer
    ingest_turn(persona_dir, {"session_id": "orphan", "speaker": "user", "text": "hi"})
    ingest_turn(persona_dir, {"session_id": "orphan", "speaker": "assistant", "text": "hello"})

    drained_sessions: list = []

    def fake_drain(persona_dir, **kw):
        drained_sessions.append(kw.get("silence_minutes"))
        return []

    monkeypatch.setattr("brain.bridge.daemon.close_stale_sessions", fake_drain)

    daemon.run_recovery_if_needed(persona_dir)
    assert 0 in drained_sessions or 0.0 in drained_sessions


def test_clean_shutdown_skips_recovery(persona_dir: Path, monkeypatch):
    from brain.bridge import daemon, state_file

    s = state_file.BridgeState(
        persona=persona_dir.name,
        pid=None,
        port=None,
        started_at="2026-04-28T10:00:00Z",
        stopped_at="2026-04-28T10:30:00Z",
        shutdown_clean=True,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)

    called = []
    monkeypatch.setattr(
        "brain.bridge.daemon.close_stale_sessions",
        lambda persona_dir, **kw: called.append(kw.get("silence_minutes")) or [],
    )

    daemon.run_recovery_if_needed(persona_dir)
    assert called == []  # not called on clean shutdown
```

- [ ] **Step 2: Run; confirm failures**

```bash
uv run pytest tests/bridge/test_lifecycle.py::test_dirty_shutdown_drains_orphan_buffers tests/bridge/test_lifecycle.py::test_clean_shutdown_skips_recovery -v
```

Expected: `ModuleNotFoundError: No module named 'brain.bridge.daemon'`.

- [ ] **Step 3: Implement `brain/bridge/daemon.py` (recovery + start/stop/status core)**

```python
"""SP-7 bridge daemon orchestration — process spawn, stop, status, recovery.

Public surface used by CLI handlers in brain.cli:
  cmd_start(args) -> int
  cmd_stop(args)  -> int
  cmd_status(args) -> int
  cmd_tail(args)  -> int

Internal:
  run_recovery_if_needed(persona_dir) — drain orphan buffers if previous bridge
    exited dirty.
  spawn_detached(persona_dir, idle_shutdown_seconds, client_origin) — double-fork
    and exec the server process, return immediately after bridge.json materialises.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from brain.bridge import state_file
from brain.bridge.provider import build_provider
from brain.ingest.pipeline import close_stale_sessions
from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

LOCKFILE = "bridge.json.lock"


# ---------------------------------------------------------------------------
# Dirty-shutdown recovery
# ---------------------------------------------------------------------------


def run_recovery_if_needed(persona_dir: Path) -> int:
    """If previous bridge exited dirty, drain orphan buffers. Returns drained count."""
    if not state_file.recovery_needed(persona_dir):
        return 0
    prev = state_file.read(persona_dir)
    logger.warning(
        "previous bridge exited dirty (pid=%s started_at=%s) — running recovery",
        prev.pid if prev else "?",
        prev.started_at if prev else "?",
    )
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    embeddings = EmbeddingCache()
    provider = build_provider(persona_dir)
    try:
        reports = close_stale_sessions(
            persona_dir,
            silence_minutes=0,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )
        return len(reports)
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# Lockfile (cross-platform via O_EXCL)
# ---------------------------------------------------------------------------


def acquire_lock(persona_dir: Path) -> int | None:
    """Create the lockfile atomically. Returns fd on success, None on conflict."""
    path = persona_dir / LOCKFILE
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except FileExistsError:
        # Stale-lock check
        try:
            existing_pid = int(path.read_text().strip())
            if not state_file.pid_is_alive(existing_pid):
                path.unlink()
                return acquire_lock(persona_dir)
        except (ValueError, OSError):
            pass
        return None


def release_lock(persona_dir: Path, fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        (persona_dir / LOCKFILE).unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


def spawn_detached(
    persona_dir: Path,
    idle_shutdown_seconds: float | None,
    client_origin: str,
    log_path: Path,
) -> int:
    """Spawn the bridge server in a detached process.

    Returns the child's pid. The child writes bridge.json once it's bound.
    Caller polls /health to confirm readiness.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab")
    cmd = [
        sys.executable,
        "-m",
        "brain.bridge.runner",
        "--persona-dir",
        str(persona_dir),
        "--client-origin",
        client_origin,
    ]
    if idle_shutdown_seconds is not None:
        cmd += ["--idle-shutdown-seconds", str(idle_shutdown_seconds)]

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


# ---------------------------------------------------------------------------
# CLI handlers — called from brain/cli.py
# ---------------------------------------------------------------------------


def cmd_start(args) -> int:
    from brain.paths import get_persona_dir, get_log_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    if state_file.is_running(persona_dir):
        cur = state_file.read(persona_dir)
        print(f"bridge already running on port {cur.port} (pid {cur.pid})", file=sys.stderr)
        return 2

    fd = acquire_lock(persona_dir)
    if fd is None:
        print("bridge already starting (lockfile held)", file=sys.stderr)
        return 2

    try:
        drained = run_recovery_if_needed(persona_dir)
        if drained:
            print(f"recovered from dirty shutdown — drained {drained} orphan sessions")

        log_path = get_log_dir() / f"bridge-{persona_dir.name}.log"
        idle = float(args.idle_shutdown) * 60 if args.idle_shutdown > 0 else None
        client_origin = getattr(args, "client_origin", "cli")
        pid = spawn_detached(persona_dir, idle, client_origin, log_path)

        # Poll for readiness — bridge.json appears, then /health returns 200
        deadline = time.time() + 5.0
        while time.time() < deadline:
            time.sleep(0.1)
            s = state_file.read(persona_dir)
            if s is not None and s.pid == pid and s.port:
                try:
                    r = httpx.get(f"http://127.0.0.1:{s.port}/health", timeout=1.0)
                    if r.status_code == 200:
                        print(f"bridge started on port {s.port} (pid {pid})")
                        return 0
                except httpx.HTTPError:
                    continue
        print(f"bridge spawned (pid {pid}) but /health did not respond in 5s", file=sys.stderr)
        return 1
    finally:
        release_lock(persona_dir, fd)


def cmd_stop(args) -> int:
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None or s.pid is None or not state_file.pid_is_alive(s.pid):
        print("bridge not running")
        return 0
    try:
        os.kill(s.pid, signal.SIGTERM)
    except ProcessLookupError:
        print("bridge not running")
        return 0

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        time.sleep(0.2)
        if not state_file.pid_is_alive(s.pid):
            print(f"bridge stopped (was pid {s.pid})")
            return 0
    print(f"bridge did not stop within {args.timeout}s", file=sys.stderr)
    return 1


def cmd_status(args) -> int:
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None:
        print("bridge: not running (no state file)")
        return 0
    if state_file.is_running(persona_dir):
        try:
            r = httpx.get(f"http://127.0.0.1:{s.port}/health", timeout=1.0)
            health = r.json()
            print(f"bridge: running pid={s.pid} port={s.port}")
            print(f"  uptime_s: {health['uptime_s']}")
            print(f"  sessions_active: {health['sessions_active']}")
            print(f"  supervisor: {health['supervisor_thread']}")
            print(f"  pending_alarms: {health['pending_alarms']}")
        except httpx.HTTPError as e:
            print(f"bridge: pid {s.pid} alive but /health unreachable: {e}", file=sys.stderr)
            return 1
    elif state_file.recovery_needed(persona_dir):
        print(f"bridge: previous process crashed dirty (pid {s.pid}) — next start will recover")
    else:
        print(f"bridge: stopped cleanly at {s.stopped_at}")
    return 0


def cmd_tail(args) -> int:
    """Subscribe to /events and print every event as a JSON line."""
    import json

    from websockets.sync.client import connect

    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None or not state_file.is_running(persona_dir):
        print("bridge not running", file=sys.stderr)
        return 1
    url = f"ws://127.0.0.1:{s.port}/events"
    try:
        with connect(url) as ws:
            while True:
                msg = ws.recv()
                print(msg)
    except KeyboardInterrupt:
        return 0
```

Also create `brain/bridge/runner.py` — the entrypoint the spawned process uses:

```python
"""SP-7 bridge runner — entrypoint for the spawned daemon process.

`python -m brain.bridge.runner --persona-dir <path> --client-origin <c> [--idle-shutdown-seconds N]`

Writes bridge.json with pid + chosen port, then runs uvicorn until SIGTERM.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from brain.bridge import state_file
from brain.bridge.server import build_app


def _allocate_port() -> int:
    """Bind ephemeral, read assigned port, close. Race window is tiny."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona-dir", required=True, type=Path)
    p.add_argument("--client-origin", default="cli")
    p.add_argument("--idle-shutdown-seconds", type=float, default=None)
    args = p.parse_args()

    persona_dir = args.persona_dir
    port = _allocate_port()

    initial_state = state_file.BridgeState(
        persona=persona_dir.name,
        pid=os.getpid(),
        port=port,
        started_at=datetime.now(UTC).isoformat(),
        stopped_at=None,
        shutdown_clean=False,
        client_origin=args.client_origin,
    )
    state_file.write(persona_dir, initial_state)

    def _on_sigterm(_signum, _frame):
        # Lifespan __aexit__ runs first; here we just confirm clean.
        pass

    signal.signal(signal.SIGTERM, _on_sigterm)

    app = build_app(
        persona_dir=persona_dir,
        client_origin=args.client_origin,
        idle_shutdown_seconds=args.idle_shutdown_seconds,
    )
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        # Lifespan teardown already ran. Mark clean.
        cur = state_file.read(persona_dir)
        if cur is not None:
            cur.pid = None
            cur.port = None
            cur.stopped_at = datetime.now(UTC).isoformat()
            cur.shutdown_clean = True
            state_file.write(persona_dir, cur)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Wire `nell bridge` subcommands into `brain/cli.py`**

In `brain/cli.py`, find where other subcommand groups are added (e.g., near the `interest` block around line 1043) and add:

```python
    # nell bridge — SP-7 daemon control
    b_sub = subparsers.add_parser(
        "bridge",
        help="Manage the per-persona bridge daemon (SP-7).",
    )
    b_actions = b_sub.add_subparsers(dest="action", required=True)

    b_start = b_actions.add_parser("start", help="Start the bridge daemon.")
    b_start.add_argument("--persona", required=True)
    b_start.add_argument(
        "--idle-shutdown",
        type=float,
        default=30,
        help="Idle-shutdown threshold in minutes (0 = never).",
    )
    b_start.add_argument(
        "--client-origin", default="cli", choices=["cli", "tauri", "tests"]
    )
    from brain.bridge.daemon import cmd_start, cmd_stop, cmd_status, cmd_tail

    b_start.set_defaults(func=cmd_start)

    b_stop = b_actions.add_parser("stop", help="Stop the bridge daemon.")
    b_stop.add_argument("--persona", required=True)
    b_stop.add_argument("--timeout", type=float, default=180.0)
    b_stop.set_defaults(func=cmd_stop)

    b_status = b_actions.add_parser("status", help="Show bridge daemon status.")
    b_status.add_argument("--persona", required=True)
    b_status.set_defaults(func=cmd_status)

    b_tail = b_actions.add_parser("tail-events", help="Tail /events as JSON lines.")
    b_tail.add_argument("--persona", required=True)
    b_tail.set_defaults(func=cmd_tail)
```

- [ ] **Step 5: Modify `nell chat` to auto-spawn-and-route via the bridge**

Find the existing `nell chat` handler in `brain/cli.py` and the parser. Add two flags:

```python
    chat_p.add_argument("--no-bridge", action="store_true",
                        help="Bypass the bridge daemon and call engine.respond() in-process.")
    chat_p.add_argument("--bridge-only", action="store_true",
                        help="Error out if bridge isn't running; do not auto-spawn.")
```

In the chat handler, near the top, add bridge-routing:

```python
def _chat_handler(args) -> int:
    from brain.paths import get_persona_dir
    from brain.bridge import state_file, daemon

    persona_dir = get_persona_dir(args.persona)
    if args.no_bridge:
        return _chat_direct_mode(args, persona_dir)  # existing logic, refactored

    if not state_file.is_running(persona_dir):
        if args.bridge_only:
            print("bridge not running (--bridge-only set)", file=sys.stderr)
            return 1
        # Auto-spawn
        class _StartArgs:
            persona = args.persona
            idle_shutdown = 30
            client_origin = "cli"
        rc = daemon.cmd_start(_StartArgs())
        if rc != 0:
            return rc

    return _chat_via_bridge(args, persona_dir)
```

`_chat_via_bridge` opens `WS /stream/{sid}` (creating a session via `POST /session/new` if `--session` not provided) and prints reply chunks as they arrive. Use `websockets.sync.client`. Implementation pattern:

```python
def _chat_via_bridge(args, persona_dir: Path) -> int:
    import json
    import sys

    import httpx
    from websockets.sync.client import connect

    from brain.bridge import state_file

    s = state_file.read(persona_dir)
    base = f"http://127.0.0.1:{s.port}"
    if args.session:
        sid = args.session
    else:
        sid = httpx.post(f"{base}/session/new", json={"client": "cli"}).json()["session_id"]

    print(f"chat session {sid} (Ctrl-D to exit)")
    while True:
        try:
            line = input("you: ").strip()
        except EOFError:
            break
        if not line:
            continue
        with connect(f"ws://127.0.0.1:{s.port}/stream/{sid}") as ws:
            ws.send(json.dumps({"message": line}))
            print("nell: ", end="", flush=True)
            while True:
                msg = json.loads(ws.recv())
                if msg.get("type") == "reply_chunk":
                    print(msg["text"], end="", flush=True)
                elif msg.get("type") == "done":
                    print()
                    break
                elif msg.get("type") == "error":
                    print(f"\n[error: {msg.get('detail', msg.get('code'))}]", file=sys.stderr)
                    return 1
    # Exit: flush via /sessions/close
    httpx.post(f"{base}/sessions/close", json={"session_id": sid})
    return 0
```

- [ ] **Step 6: Run all tests**

```bash
uv run pytest tests/bridge/ -v
```

Expected: ~25 pass.

- [ ] **Step 7: Smoke-test the full lifecycle end-to-end**

```bash
# 1. Start
uv run nell bridge start --persona nell.sandbox --idle-shutdown 0

# 2. Status
uv run nell bridge status --persona nell.sandbox

# 3. Chat (auto-routes via bridge)
echo -e "hi\n" | uv run nell chat --persona nell.sandbox

# 4. Tail events in another terminal
uv run nell bridge tail-events --persona nell.sandbox &
sleep 1; kill %1

# 5. Force-kill the daemon (simulates crash)
PID=$(uv run python -c "from brain.bridge import state_file; from brain.paths import get_persona_dir; s = state_file.read(get_persona_dir('nell.sandbox')); print(s.pid)")
kill -9 "$PID"

# 6. Verify dirty status
uv run nell bridge status --persona nell.sandbox
# Expected: "previous process crashed dirty (pid X) — next start will recover"

# 7. Restart — should run recovery
uv run nell bridge start --persona nell.sandbox --idle-shutdown 0
# Expected: prints "recovered from dirty shutdown — drained N orphan sessions" then "bridge started on port X"

# 8. Clean shutdown
uv run nell bridge stop --persona nell.sandbox

# 9. Verify clean
uv run nell bridge status --persona nell.sandbox
# Expected: "bridge: stopped cleanly at <iso>"
```

Each numbered step is a verification gate — if any fails, stop and fix before moving on.

- [ ] **Step 8: Commit**

```bash
git add brain/bridge/daemon.py brain/bridge/runner.py brain/cli.py tests/bridge/test_lifecycle.py
git commit -m "feat(sp7): nell bridge CLI + dirty-shutdown recovery + auto-spawn from chat"
```

---

## Self-Review

After all seven tasks land:

- [ ] **Run the full test suite to confirm no regressions:**

```bash
uv run pytest -v
```

Expected: every test (existing + ~25 new bridge tests) passes.

- [ ] **Confirm the spec coverage matrix:**

| Spec section | Covered by |
|---|---|
| §3 Transport (HTTP+WS) | Task 4 + Task 5 |
| §4 Architecture / file map | All tasks |
| §5 API surface — 7 endpoints | Tasks 4, 5, 6 |
| §6 State file + dirty-shutdown recovery | Tasks 3, 7 |
| §7 Lifecycle (CLI, auto-spawn, idle, Tauri contract, graceful shutdown) | Tasks 6, 7 |
| §8 Supervisor thread | Task 6 |
| §9 Event bus + 14-event catalogue | Task 2 + every task that emits |
| §10 Streaming behavior (simulated chunking) | Task 5 |
| §11 Failure modes — 12 cases | Tasks 5, 6, 7 (most have explicit tests) |
| §12 Testing — 26 tests | Tasks 2, 3, 4, 5, 6, 7 (≥25 ship) |

- [ ] **Confirm no placeholders remain:** search the working tree for `TODO`, `FIXME`, `TBD`:

```bash
grep -rn "TODO\|FIXME\|TBD" brain/bridge/ tests/bridge/
```

Expected: zero hits (the spec's reserved `outbox` payload `(TBD)` lives in the spec, not the code).

- [ ] **Re-read §11 of the spec one last time** and confirm every failure-mode row maps to either a test, a deliberate non-test (like SIGKILL recovery already covered by Task 7 step 7), or an out-of-scope row (§13).

---

## Out of Scope (Confirm Not Built)

These are deferred per spec §13 and must NOT be built in this plan:

- Multi-modal content (images/audio in chat)
- TTS voice synthesis
- `GET /tools/log` endpoint
- Token-based auth
- Event replay / persistent event log
- Outbox / proactive messaging emitter
- launchd/systemd integration
- Recovery dream on dirty-shutdown

If implementation pressure tries to add one of these, push it to a future SP-7.x spec instead.
