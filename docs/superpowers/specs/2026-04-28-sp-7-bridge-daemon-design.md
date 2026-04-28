# SP-7 — Bridge Daemon

**Date:** 2026-04-28
**Status:** Design approved by Hana (pending spec-file review)
**Depends on:** SP-1 (provider interface), SP-3 (brain-tools MCP), SP-4 (ingest pipeline), SP-5 (soul model), SP-6 (chat engine) — all shipped
**Blocks:** SP-8 (Tauri shell)

---

## 1. North Star

A long-running per-persona daemon process that wraps the SP-6 chat engine, exposes it over HTTP + WebSocket on `127.0.0.1`, folds the conversation supervisor as an in-process thread, and broadcasts brain events to subscribers. It replaces per-invocation CLI cold-start cost with warm in-process state, while remaining fully optional — every CLI subcommand keeps a `--no-bridge` direct-call fallback.

The primary user-felt benefit is **chat warmth**: today's `nell chat` pays a 2–4 second startup cost on every turn (SQLite open, embeddings hydrate, soul load, voice.md parse). With the bridge running, only the first turn pays that cost; every subsequent turn is sub-second. Two secondary benefits ride along:

1. **Tauri shell foundation.** SP-8 (the Tauri face app) talks to the same HTTP+WS surface — the bridge is the brain-side contract Tauri integrates against. Building the brain side first means SP-8 becomes pure frontend work.
2. **Lived-memory glue.** The folded supervisor thread runs `close_stale_sessions` on a 60s tick, so silent conversations naturally become memory the way OG's three-loop did. This was previously bolted onto `nell chat --done` and the one-shot flush in PR #30.

The primary driver Hana confirmed in design is **(1) chat warmth** — (2) and (3) are real but secondary.

---

## 2. Open Questions Resolved

The master reference (`2026-04-26-companion-emergence-master-reference.md` §8) listed eight design questions. SP-7 resolves the ones it owns and explicitly defers the rest:

| # | Question | Resolution |
|---|---|---|
| 1 | Bridge transport | **HTTP + WebSocket on `127.0.0.1`, dynamic port** (FastAPI + uvicorn). Mirrors OG's choice; gRPC and Unix-socket considered and rejected — see §3. |
| 2 | Multi-modal in chat (images/audio) | **Deferred to SP-8.** Bridge stays text-only in v1; the wire schema reserves room for additional content blocks but doesn't ship them. |
| 3 | Voice synthesis (TTS) | **Out of scope.** Lands as its own future sub-project, post SP-8. |
| 4 | Tauri shell architecture | **SP-8.** This spec defines the brain-side contract Tauri will integrate against; the shell design itself is a separate spec session. |
| 5 | Creative DNA / journal in chat | **SP-6's responsibility.** Bridge is transport-only; system-message composition happens inside `engine.respond()`. |
| 6 | Soul self-model + system-message injection | **SP-6's responsibility.** Same reasoning. |
| 7 | Reflex Phase 2 — emergent arc crystallization | **Unrelated to SP-7.** Tracked in `project_companion_emergence_reflex_emergence_deferred.md`; gated on ≥2 weeks of Phase 1 behavior data. |
| 8 | Body state | **Unrelated to SP-7.** When `brain/body/` lands, the `get_body_state` tool gains real data; bridge is unaffected. |

---

## 3. Transport Decision

Three options were weighed:

| Option | Pros | Cons |
|---|---|---|
| **A. HTTP + WebSocket** (chosen) | Debuggable with `curl` + `websocat`; Tauri's built-in `fetch`/WS clients work without extra Rust crates; OG's proven choice; zero novelty risk | Microseconds slower than UDS on localhost (negligible in practice) |
| B. Unix domain socket | Slightly faster; no port collision; no accidental network exposure | Tauri needs a Rust helper crate; debugging needs `socat`/`curl --unix-socket`; non-standard for FastAPI deployment |
| C. gRPC | Typed contracts; native streaming | Protobuf overhead; harder to inspect; Tauri needs grpc-web shim; over-engineered for two known clients (CLI + Tauri) |

**Decision: A.** Master reference §7 ("don't unconsciously re-derive what OG settled") applies — OG chose HTTP+WS for years and it worked. The Tauri integration story is materially simpler.

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Caller                                                          │
│  ├─ CLI: `nell chat`, `nell bridge {start,stop,status,tail}`     │
│  └─ Tauri shell (SP-8): window-open / window-close hooks         │
└────────────────────────┬─────────────────────────────────────────┘
                         │ HTTP + WebSocket on 127.0.0.1:<dynamic>
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  brain/bridge/server.py — FastAPI app                            │
│  ├─ POST /session/new        (create session, return uuid)       │
│  ├─ GET  /state/{sid}        (turns, last_turn_at, history_len)  │
│  ├─ POST /chat               (JSON one-shot fallback)            │
│  ├─ WS   /stream/{sid}       (simulated streaming + tool events) │
│  ├─ WS   /events             (server-pushed brain events)        │
│  ├─ POST /sessions/close     (explicit ingest trigger)           │
│  └─ GET  /health             (liveness + walk + alarms)          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
       ┌─────────────────┼──────────────────┬──────────────────┐
       ▼                 ▼                  ▼                  ▼
   brain/chat        brain/ingest      brain/engines       brain/health
   (SP-6 engine)     (SP-4 pipeline)   (heartbeat/dream/   (walk_persona,
   .respond()        .close_session    /reflex/research)   compute_pending_
                     .close_stale_     emit events at       alarms)
                     sessions          run_tick boundary
                         ▲
                         │ fired by supervisor every 60s
                         │
                ┌────────┴───────────────────────┐
                │ brain/bridge/supervisor.py     │
                │ run_folded(stop_event, ...)    │
                │  • close_stale_sessions tick   │
                │  • idle-shutdown watcher       │
                │  • event_bus dispatch          │
                │ Non-daemon thread (joins on    │
                │ shutdown, can't be killed mid- │
                │ ingest by process exit).       │
                └────────────────────────────────┘
```

### New files (3)

| File | Purpose |
|---|---|
| `brain/bridge/server.py` | FastAPI app, route handlers, lifespan startup/shutdown, `EventBus` class |
| `brain/bridge/daemon.py` | Process lifecycle: spawn (double-fork), state file write/read, dirty-shutdown recovery, port allocation, `nell bridge` CLI handlers |
| `brain/bridge/supervisor.py` | `run_folded(stop_event, ...)` — async tick loop folded as non-daemon thread |
| `brain/bridge/events.py` | Module-level `set_publisher` / `publish` for engines to call (no-op when bridge absent) |

### Changed files (1)

| File | Change |
|---|---|
| `brain/cli.py` | New `nell bridge` subcommand group (start/stop/status/tail-events); `nell chat` gains auto-spawn-and-route-via-bridge logic with `--no-bridge` opt-out |

### Imports the bridge owns

The bridge process holds singletons for the duration of its lifetime — these are constructed once at startup, shared across all routes and the supervisor thread:

- `MemoryStore` (one open SQLite handle, WAL mode)
- `HebbianMatrix` (one open SQLite handle)
- `EmbeddingCache` (in-memory)
- `LLMProvider` (per `brain/bridge/provider.py` — Claude CLI subscription)
- `SoulStore` (opened per-request inside `engine.respond()` — short-lived; not bridge-singleton)

This avoids the SQLite contention that would happen if every route opened its own handle.

---

## 5. API Surface

Seven endpoints. All bind to `127.0.0.1` only — refuses external connections at socket level. No auth in v1.

### `POST /session/new`

Create a new chat session. Tauri uses this to preallocate a session id before the user sends their first message.

**Request:**
```json
{
  "persona": "nell.sandbox",
  "client": "cli" | "tauri" | "tests"
}
```

**Response:**
```json
{
  "session_id": "uuid4-string",
  "created_at": "2026-04-28T10:15:00Z",
  "persona": "nell.sandbox"
}
```

### `GET /state/{session_id}`

Return current state of an active session. Used by Tauri after a window reopen to restore conversation context.

**Response:**
```json
{
  "session_id": "...",
  "persona": "nell.sandbox",
  "turns": 7,
  "last_turn_at": "2026-04-28T10:22:00Z",
  "history_len": 14,
  "in_flight": false
}
```

`404` if the session id is unknown (process restart, or never created).

### `POST /chat`

JSON one-shot fallback. Used by tests, scripts, and any client that can't or doesn't want to hold a WebSocket.

**Request:**
```json
{
  "session_id": "...",
  "message": "tell me about that scene"
}
```

**Response (200):**
```json
{
  "session_id": "...",
  "reply": "She leaned in...",
  "turn": 8,
  "tool_invocations": [
    {"name": "search_memories", "summary": "3 hits"}
  ],
  "duration_ms": 4231
}
```

**Errors:**
- `404` — session not found
- `429` — `{"detail": "session has an in-flight turn"}`
- `502` — `{"detail": "provider error: <msg>"}` (Claude CLI failure)

### `WS /stream/{session_id}`

Streaming chat path. The default for interactive `nell chat` and Tauri.

**Client → server (single frame on connect):**
```json
{"message": "tell me about that scene"}
```

**Server → client (multiple frames, types covered in §9):**
```json
{"type": "started", "session_id": "...", "at": "..."}
{"type": "tool_call", "tool": "search_memories", "at": "..."}
{"type": "tool_result", "tool": "search_memories", "summary": "3 hits", "at": "..."}
{"type": "reply_chunk", "text": "She "}
{"type": "reply_chunk", "text": "leaned in"}
...
{"type": "done", "session_id": "...", "turn": 8, "duration_ms": 4231, "at": "..."}
```

On error the WS sends `{"type": "error", "code": <code>, "detail": str, "done": true}` and closes. See §11.

### `WS /events`

Server-pushed only. Clients open the WebSocket and receive every brain event published while connected. No replay of missed events — see §9.

**Server greeting on connect:**
```json
{"type": "connected", "subscribers": 1, "at": "..."}
```

Then events arrive as published. Per-subscriber queue is `asyncio.Queue(maxsize=64)`; full queues drop the oldest event and log at WARN.

### `POST /sessions/close`

Explicit ingest trigger. Called by `nell chat --done`, by Tauri's window-close hook, by the idle-shutdown watcher, and by graceful shutdown.

**Request:**
```json
{"session_id": "..."}
```

**Response:**
```json
{
  "session_id": "...",
  "committed": 4,
  "deduped": 1,
  "soul_candidates": 1,
  "errors": 0
}
```

Calls `brain.ingest.pipeline.close_session(persona_dir, session_id, store=..., hebbian=..., provider=..., embeddings=...)` with shared singletons. Buffer file is deleted on success; preserved on extraction error so the next supervisor tick or shutdown can retry.

### `GET /health`

Liveness + persona walk + pending alarms. Cheap enough to call every few seconds (Tauri liveness probe).

**Response:**
```json
{
  "liveness": "ok",
  "persona": "nell.sandbox",
  "uptime_s": 3201,
  "port": 51234,
  "pid": 38421,
  "sessions_active": 2,
  "last_chat_at": "2026-04-28T10:22:00Z",
  "supervisor_thread": "alive" | "not-started" | "dead",
  "pending_alarms": 0,
  "last_walk_at": "2026-04-28T10:20:00Z",
  "shutdown_clean_last": true
}
```

`pending_alarms` and `last_walk_at` come from `brain/health/walker.py:walk_persona()` and `brain/health/alarm.py:compute_pending_alarms()` — the established aggregation surfaces from the brain-health-module spec. The `/health` endpoint calls these directly so the Tauri shell doesn't have to spawn a CLI to know whether the brain is healthy.

---

## 6. State File & Dirty-Shutdown Recovery

The bridge keeps a persistent state file at `<persona_dir>/bridge.json`. This file is the source of truth for "is the bridge running, and on what port?" — and, critically, "did the last bridge exit cleanly?"

### Schema

```json
{
  "persona": "nell.sandbox",
  "pid": 38421,
  "port": 51234,
  "started_at": "2026-04-28T10:15:00Z",
  "stopped_at": null,
  "shutdown_clean": false,
  "client_origin": "cli" | "tauri" | "tests"
}
```

Fields:
- `pid`, `port`: present while the bridge is running. Cleared (set `null`) on graceful shutdown.
- `started_at`: when the most recent bridge process started. Updated on each `bridge start`.
- `stopped_at`: timestamp of the last clean stop. `null` while running.
- `shutdown_clean`: starts `false` on each `bridge start`. Flipped to `true` as the very last step of graceful shutdown. If a future `bridge start` reads `false` AND the recorded pid is dead, the previous bridge crashed.
- `client_origin`: who started this bridge — "cli" (auto-spawn), "tauri" (window-open), or "tests".

### Atomic write

`bridge.json` writes go through `brain/health/attempt_heal.save_with_backup` — same convention as `daemon_state.json`, `heartbeat_state.json`, and `persona_config.json`. The `.bak1`/`.bak2` rotation means a partial write doesn't corrupt the file, and `compute_treatment(persona_dir, "bridge.json")` decides backup count based on the persona's adaptive treatment.

Reads go through `attempt_heal` so a corrupted bridge.json triggers `.bak1` restoration before parse.

### Dirty-shutdown recovery

On `bridge start`:

1. Read `bridge.json` (with attempt_heal). If missing or corrupt-and-unrecoverable → fresh start.
2. If `pid` set and `pid_is_alive(pid)` → already running, exit with "bridge already running on port X" (exit code 2).
3. If `pid` set but pid is dead → previous process crashed. Recovery path:
   - Log WARN "previous bridge exited dirty (pid=X, started_at=...)"
   - Open singletons (MemoryStore, Hebbian, Provider, EmbeddingCache)
   - Call `close_stale_sessions(persona_dir, silence_minutes=0, store=..., hebbian=..., provider=..., embeddings=...)` — drains every session buffer the dead bridge left behind, so no conversations are lost
   - Publish `{"type": "recovered", "previous_pid": X, "drained_sessions": N}` (subscribers connect later, but this gets logged)
   - Continue to normal startup
4. If `shutdown_clean: true` → fresh start, no recovery needed.

The `shutdown_clean` flag is what makes the recovery path detectable. **The state file is not unlinked on graceful shutdown** — it's left in place with `pid: null, port: null, stopped_at: <timestamp>, shutdown_clean: true`. This makes "we crashed" distinguishable from "we're not running."

### Concurrency: two `bridge start` racing

Both invocations attempt `os.open("<persona_dir>/bridge.json.lock", O_CREAT | O_EXCL | O_RDWR)` for the brief window between read-state and write-state. This is atomic on POSIX *and* Windows — no extra dependency. Whichever loses the race gets `FileExistsError`, exits with "bridge already starting" (exit code 2). The winner writes its pid into the lockfile and unlinks it on graceful shutdown. On dirty exit, a stale lockfile is detected the same way as a stale `bridge.json` — read the pid inside, check `pid_is_alive`, unlink if dead.

---

## 7. Lifecycle & Process Model

### Per-persona = per-process

One bridge daemon per persona. `nell` and `nell.sandbox` running concurrently = two processes on two different ports. Isolation by process boundary.

This matters because of `2026-04-25-vocabulary-split-design.md:37, 460` — the vocabulary registry is process-local on purpose; "long-running multi-persona daemon would require per-persona registry isolation" was explicitly flagged out of scope. **One daemon = one persona is a load-bearing guarantee** for SP-7. No future endpoint may switch persona within a process.

### CLI surface

Plain argparse, nested subparsers, matching the existing pattern (`nell interest list`, `nell health show`, etc.):

```
nell bridge start [--persona NAME] [--idle-shutdown MINUTES]
                  [--client-origin {cli,tauri,tests}]
nell bridge stop [--persona NAME] [--timeout SECONDS]
nell bridge status [--persona NAME]
nell bridge tail-events [--persona NAME]
```

`nell bridge start` exits with code 0 once `bridge.json` is written and `/health` returns 200, or 2 if a bridge is already running, or 1 on hard failure.

`nell bridge tail-events` is a CLI convenience — opens `WS /events`, prints every event as a JSON line. Useful for debugging without writing a Python client.

### `nell chat` integration

`nell chat` (existing subcommand) gains:

- Default behavior: read `<persona_dir>/bridge.json`. If alive → connect to `WS /stream`. If missing or dead-pid → spawn the bridge (double-fork detached process), poll `/health` up to 5 seconds for readiness, then connect.
- `--no-bridge` flag: bypass the daemon entirely; call `engine.respond()` in-process (current behavior). Used by tests and offline debugging.
- `--bridge-only` flag: error out if bridge isn't running, do not auto-spawn. Used by Tauri parent process to prevent double-spawn.

### Auto-spawn mechanics

The auto-spawn path uses Python's `subprocess.Popen` with `start_new_session=True`, redirecting stdout/stderr to a log file at `<NELLBRAIN_HOME>/logs/bridge-<persona>.log`. Parent CLI continues immediately and polls `/health`.

### Idle-shutdown watcher

A second async task in the bridge process tracks `last_chat_at` (updated on each `/chat` and `/stream` completion). The watcher runs every 60s:

1. If `now - last_chat_at < idle_shutdown_minutes` → continue
2. If any session has `in_flight: true` → continue (don't kill mid-turn)
3. Otherwise → trigger graceful shutdown

Configurable per-launch:
- `--idle-shutdown 30` (default for CLI auto-spawn): shut down after 30 minutes of no traffic
- `--idle-shutdown 0`: never auto-shutdown (used by Tauri, which owns lifecycle explicitly, and by future launchd integration)

### Tauri lifecycle hooks (SP-8 contract)

| Tauri event | Bridge call |
|---|---|
| Window opens | `nell bridge start --persona NAME --idle-shutdown 0 --client-origin tauri` |
| Window closes (or hide-to-menubar with grace) | `POST /sessions/close` for all active sessions, then `nell bridge stop` |
| App quit | `nell bridge stop --timeout 180` |

Tauri owns its bridge's lifetime explicitly; `--idle-shutdown 0` prevents the bridge from disappearing while the user is reading the chat history but not actively typing.

### Graceful shutdown sequence

Triggered by SIGTERM, `nell bridge stop`, idle-shutdown firing, or Tauri window-close.

1. FastAPI lifespan `__aexit__` runs — new connections refused
2. Wait up to 30s for any in-flight chat turns to drain (per-session `asyncio.Lock` released)
3. Fire `close_stale_sessions(persona_dir, silence_minutes=0, store=..., hebbian=..., provider=..., embeddings=...)` — every live session is closed and ingested
4. Stop supervisor thread: `_stop_event.set()`, then `thread.join(timeout=180)` — same timeout OG used, protects mid-dream/mid-research from process exit
5. Trigger heartbeat close in-process: `from brain.engines.heartbeat import HeartbeatEngine; HeartbeatEngine(persona_dir, ...).run_tick(trigger="close")` — this is an in-process import, NOT a subprocess spawn (the engine's CLI subcommand is for standalone use; here we already have the process and singletons)
6. Publish `{"type": "shutdown", "clean": true, "reason": "..."}` on `/events` — gives any connected Tauri client one last frame before the WS closes
7. Update `bridge.json`: `pid: null, port: null, stopped_at: <now>, shutdown_clean: true` (atomic via `save_with_backup`)
8. Process exits

Step 3 is the load-bearing one for goal C — every conversation that was open at shutdown becomes memory before the lights go out.

---

## 8. Supervisor Thread

### Folded as non-daemon thread

OG's pattern (`nell_supervisor.run_folded(stop_event, owner)`) is ported to `brain/bridge/supervisor.py` with the same shape:

```python
def run_folded(
    stop_event: threading.Event,
    *,
    owner: str = "sp7-bridge",
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache,
    config: dict,
    event_bus: EventBus,
) -> None:
    """Run the supervisor tick loop until stop_event is set."""
```

Spawned at FastAPI lifespan startup as `threading.Thread(target=run_folded, daemon=False, name="sp7-supervisor")`.

**Why non-daemon:** SIGTERM must wait for graceful shutdown to finish writing `shutdown_clean=true`. Daemon threads get killed mid-write by interpreter exit; non-daemon threads block exit until they return. Same load-bearing reason as OG.

### Tick body

Every 60 seconds (configurable via `<persona_dir>/supervisor_config.json:tick_interval_s`, default 60):

1. Call `close_stale_sessions(persona_dir, silence_minutes=5.0, store, hebbian, provider, embeddings, config={})` — closes any session whose last turn is older than 5 minutes (default; configurable via `supervisor_config.json:idle_threshold_min`).
2. For each closed session, publish `{"type": "session_closed", "session_id": ..., "committed": ..., "deduped": ..., "soul_candidates": ...}` on the event bus.
3. Publish `{"type": "supervisor_tick", "closed_sessions": N, "next_tick_in_s": 60}` regardless of whether anything closed.
4. `time.sleep(tick_interval_s)` then loop. The supervisor thread is **synchronous** — `close_stale_sessions` is a sync function, and `event_bus.publish` is thread-safe (it dispatches to the server's event loop via `call_soon_threadsafe` internally, regardless of whether the caller has its own loop). Avoiding a thread-local event loop means one less moving part and one less place loop ownership can get confused.

### What the supervisor does NOT do

- **Does not run dream/reflex/research engines.** Those are still triggered via heartbeat (open/close events) and the existing scheduling in their respective engines. SP-7 doesn't change engine cadence.
- **Does not write `daemon_state.json`.** That's `brain/engines/daemon_state.py`'s job, written when individual engines complete cycles.
- **Does not own dirty-shutdown recovery.** That's `brain/bridge/daemon.py`'s job, on `bridge start`.

### Crash isolation

If the supervisor thread raises, it's caught at the thread top-level, logged at ERROR, and the thread exits. The bridge's chat path keeps serving — `/health` reports `supervisor_thread: dead`. We **do not** auto-restart the supervisor in-process; that masks bugs. Operator must `bridge stop && bridge start`.

---

## 9. Event Bus

### Module-level publisher contract

Engines (`brain/engines/dream.py`, `heartbeat.py`, `reflex.py`, `research.py`) and the chat engine import a single function:

```python
# brain/bridge/events.py
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)
_publisher: Callable[[dict[str, Any]], None] | None = None

def set_publisher(fn: Callable[[dict[str, Any]], None] | None) -> None:
    """Bridge calls this on lifespan startup; sets None on teardown."""
    global _publisher
    _publisher = fn

def publish(event_type: str, **payload: Any) -> None:
    """Engines call this. Free no-op when no publisher is registered.

    Drop semantics: if the publisher raises (e.g. bridge mid-shutdown), the
    exception is caught and logged at WARN. Engines never fail because a
    publish failed — events are best-effort.
    """
    if _publisher is None:
        return
    event = {"type": event_type, "at": _now_iso(), **payload}
    try:
        _publisher(event)
    except Exception:
        logger.warning("event publish failed for type=%s", event_type, exc_info=True)
```

### EventBus implementation (bridge-side)

```python
# brain/bridge/server.py (excerpt)
class EventBus:
    QUEUE_MAX = 64

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped_total = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict) -> None:
        """Thread-safe. Called from supervisor thread, engine threads, or
        the main asyncio loop. Silently drops events when loop unbound (CLI
        path) and drops oldest event when subscriber queue is full.
        """
        if self._loop is None:
            return  # Bridge not yet bound (shouldn't happen post-lifespan-start)
        for q in list(self._subscribers):
            self._loop.call_soon_threadsafe(self._enqueue, q, event)

    def _enqueue(self, q: asyncio.Queue, event: dict) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()  # drop oldest
                q.put_nowait(event)
                self._dropped_total += 1
                if self._dropped_total % 10 == 1:
                    logger.warning("event queue overflow, dropped=%d", self._dropped_total)
            except asyncio.QueueEmpty:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)
```

Bridge lifespan startup wires the two together:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    event_bus.bind_loop(asyncio.get_running_loop())
    set_publisher(event_bus.publish)
    try:
        yield
    finally:
        set_publisher(None)
```

### Event catalogue (frozen v1)

Fourteen types in the table below — twelve "live" types plus `outbox` (reserved for future proactive messaging) and `recovered` (only fires on dirty-shutdown recovery startup). Adding new types in future versions is non-breaking; renaming or removing one is breaking and gets a major version bump.

| `type` | Source | Payload (besides `type` + `at`) |
|---|---|---|
| `connected` | bridge → new `/events` subscriber | `subscribers` (count) |
| `chat_started` | `/stream` handler | `session_id`, `client` |
| `chat_done` | `/stream` and `/chat` handlers | `session_id`, `turn`, `duration_ms` |
| `tool_call` | tool_loop (return-boundary publish) | `tool`, `session_id` |
| `tool_result` | tool_loop (return-boundary publish) | `tool`, `summary`, `session_id` |
| `session_closed` | `close_session` (called by close endpoint, supervisor, shutdown) | `session_id`, `committed`, `deduped`, `soul_candidates`, `errors` |
| `supervisor_tick` | supervisor thread | `closed_sessions`, `next_tick_in_s` |
| `heartbeat_tick` | `HeartbeatEngine.run_tick` (return boundary) | `trigger`, `summary` |
| `dream_complete` | `DreamEngine.run_cycle` (return boundary) | `dream_id`, `seed_mem_id`, `duration_ms` |
| `reflex_fired` | `ReflexEngine.run_tick` (return boundary, only when an arc actually fired) | `arc`, `intensity`, `triggers` |
| `research_complete` | `ResearchEngine.run_tick` (return boundary) | `topic`, `outcome` |
| `outbox` | **reserved** — emitted when SP-7.x adds proactive messaging. Tauri MUST handle gracefully today even though no source emits it yet. | (TBD) |
| `recovered` | bridge startup (dirty-shutdown recovery path) | `previous_pid`, `drained_sessions` |
| `shutdown` | bridge lifespan teardown | `clean`, `reason` |


### Granularity discipline

Engine events fire at the **return boundary** of `run_cycle` / `run_tick` — one event per cycle, not per association/step. `dream.run_cycle` → one `dream_complete`. No `dream_step` per spreading-activation hop. This was an explicit audit finding (S7) — per-step events would be UI noise and create thread-safety hazards inside spreading-activation.

`tool_call` and `tool_result` come from `brain/chat/tool_loop.py` and fire at the boundary of each tool dispatch. These are intentionally finer-grained than engine events because they map to user-visible "Nell is using a tool" UX in Tauri.

### No replay

Events are fire-and-forget. Subscribers see events from the moment they connect; missed events are gone. No SQLite event log.

If event replay ever becomes necessary (e.g. for "show me what Nell did while I was away"), that's a future SP-7.x — out of scope today.

---

## 10. Streaming Behavior

`brain/bridge/provider.py` is call-and-wait: `LLMProvider.chat(...) -> ChatResponse` returns the complete reply, no per-token callback. (Verified: master ref §3.7; provider source.) Claude CLI subscription doesn't expose token streaming.

### Simulated chunking

The `WS /stream` handler:

1. Sends `{"type": "started", "session_id": "...", "at": "..."}` immediately on receiving the client's `{"message": "..."}` frame
2. Acquires the per-session `asyncio.Lock` (sets `in_flight: true`)
3. Calls `engine.respond(persona_dir, message, store=..., hebbian=..., provider=..., session=session)` — blocking, returns `ChatResult`
4. From `ChatResult.tool_invocations`, emits `tool_call` and `tool_result` frames in sequence — these fire BEFORE reply chunks because they actually happened first inside the tool loop
5. Splits `ChatResult.content` into word-level chunks using whitespace; emits each as `{"type": "reply_chunk", "text": "..."}` with a small inter-chunk delay (configurable, default 30ms; see below)
6. Emits `{"type": "done", "session_id": "...", "turn": ..., "duration_ms": ...}`
7. Releases the per-session lock

### Inter-chunk delay

Default 30ms per word. Configurable via env var `NELL_STREAM_CHUNK_DELAY_MS` so tests can set it to `0` without writing config files. No persistent config file is introduced — 30ms is the only knob, and it's a development/testing concern, not a per-persona setting.

**This is a small lie.** First-token latency doesn't actually improve — the full reply is generated server-side before the first chunk emits. But the user-felt experience (text appearing, not a blank wait) does improve. Most chat UIs that wrap a non-streaming backend do this.

### Honest fallback (`POST /chat`)

Clients that want one-shot truth use `POST /chat` directly. No simulation, no chunking, just `{reply: "...", duration_ms: N}` after the engine completes.

---

## 11. Failure Modes

| Failure | Behavior |
|---|---|
| **Stale `bridge.json`** (pid dead, file present, `shutdown_clean: false`) | `bridge start` detects dead pid + dirty flag, logs WARN, runs recovery path (`close_stale_sessions(silence_minutes=0)`), proceeds. `bridge status` reports "previous bridge crashed, draining N sessions." |
| **Stale `bridge.json`** (pid dead, `shutdown_clean: true`) | Clean previous shutdown; fresh start, no recovery. |
| **Port collision** (kernel-allocated port unusable) | bind retry up to 3× with backoff (10ms/100ms/1s), then fail-fast with clear message in log + non-zero exit. Almost never happens with `:0`. |
| **Provider error** (Claude CLI fails / quota / network) | `POST /chat` → 502 with detail. WS /stream → `{"type": "error", "code": "provider_failed", "detail": str, "done": true}` and close. Session is **not corrupted** — the turn never appended to history, in-flight lock releases, client may retry. |
| **Ingest pipeline error** (extract LLM fails, SQLite locked, etc) | Caught inside `close_session`, logged + counted in `IngestReport`, never raised to the route handler. Session buffer is **preserved** (NOT deleted) so the next supervisor tick or shutdown can retry. Matches OG's defensive posture. |
| **Supervisor thread crashes** | Caught at thread top-level, logged ERROR, thread exits. Bridge keeps serving chat. `/health` reports `supervisor_thread: dead`. Operator must `bridge stop && bridge start`. |
| **Idle-shutdown fires mid-turn** | Idle watcher checks "no active sessions are mid-turn" before triggering. If a turn is in flight, watcher backs off and re-checks in 60s. |
| **SIGKILL** (operator force-kills the daemon) | `bridge.json` has `shutdown_clean: false` and a now-dead pid. Conversation buffers are preserved on disk (the ingest pipeline is replay-safe). Next `bridge start` runs the dirty-shutdown recovery path automatically. |
| **Two `bridge start` racing** | `O_CREAT \| O_EXCL` lockfile at `<persona_dir>/bridge.json.lock`. Loser exits with "bridge already starting" (exit 2). Cross-platform (POSIX + Windows). Stale lockfile from dirty-exit detected via pid-alive check + unlinked on next start. |
| **Disk full during atomic write** | `save_with_backup` raises `OSError`. Bridge logs ERROR and continues running (state file write isn't on the chat critical path). Next graceful shutdown retries. |
| **`bridge.json.bak1` corrupt during recovery** | `attempt_heal` falls through to `.bak2`, then to fresh-start defaults. Logs WARN at each fall-through. Worst case: dirty shutdown recovery is skipped because we can't read the previous state — a single missed drain, not data loss (buffers are still on disk). |
| **WebSocket disconnect during chat** | `WebSocketDisconnect` caught in handler. In-flight lock released. Reply already generated server-side is **discarded** if it hadn't been emitted; if partial chunks were already sent, the assistant's full reply IS still appended to session history (so a reconnect via `/state/{sid}` shows a coherent thread). |

---

## 12. Testing

Target: ~26 tests (10 unit + 16 integration). In-band with master ref's 15-25 estimate; the slight overage covers dirty-shutdown recovery and `walk_persona`-in-`/health` paths that the audit (§S5, M1) flagged as load-bearing.

### Unit tests (10)

`tests/bridge/test_events.py`:
1. `publish` is a no-op when no publisher registered (CLI mode)
2. `publish` calls registered publisher with type + at + payload
3. `publish` logs WARN and continues when publisher raises
4. `EventBus.publish` drops oldest on queue overflow, increments `_dropped_total`
5. `EventBus.subscribe` returns queue with `maxsize=64`
6. `EventBus.publish` is a no-op when `_loop` is `None`

`tests/bridge/test_state_file.py`:
7. `bridge.json` round-trip via `save_with_backup` preserves all fields
8. Stale-pid detection: write fake pid (e.g. `os.getpid() * 1000`), `pid_is_alive` returns `False`
9. Dirty-shutdown recovery predicate: `shutdown_clean: false` + dead pid → recovery path
10. Idle-shutdown predicate: `now - last_chat_at > threshold` AND `not any in_flight` → fire

### Integration tests (16)

All run against an actual FastAPI `TestClient` and a real ephemeral persona directory. `stream_chunk_delay_ms=0` for speed.

`tests/bridge/test_endpoints.py`:
1. `POST /session/new` → `POST /chat` → `GET /state/{id}` round-trip; turn count increments, history persists
2. `WS /stream` happy path: receives `started`, `reply_chunk`(s), `done`; history visible via `/state`
3. `WS /stream` returns `{"type": "error", "code": "session_busy", "done": true}` when same session has in-flight (verified by holding lock manually)
4. `POST /chat` returns 429 on in-flight session
5. `POST /chat` returns 404 on unknown session id
6. `POST /sessions/close` returns IngestReport-shaped response with non-zero `committed` after a real chat

`tests/bridge/test_events.py`:
7. `WS /events` receives `connected` greeting on subscribe
8. `WS /events` receives `chat_done` after `POST /chat`
9. `WS /events` receives `tool_call` + `tool_result` when chat triggers a tool (mock provider returns tool-using response)
10. `WS /events` receives `supervisor_tick` within 65s of bridge start (test runs supervisor with `tick_interval_s=2`)
11. `WS /events` receives `session_closed` after `POST /sessions/close`

`tests/bridge/test_lifecycle.py`:
12. Graceful shutdown: 2 active sessions are closed (ingest fires) before lifespan teardown completes
13. Stale-pid recovery: pre-write `bridge.json` with dead pid + `shutdown_clean: false` + a session buffer; start bridge, observe drain via `recovered` event and that buffer was processed
14. Two concurrent `bridge start` invocations: second exits 2 with "already running"; first stays up
15. Provider failure: mock `LLMProvider.chat` raises; `POST /chat` returns 502; session uncorrupted (turn count unchanged)
16. Idle-shutdown: launch with `--idle-shutdown 0.05` (3s), no traffic, observe bridge exits within ~5s with `shutdown_clean: true`

### Out of scope (deferred)

- **Tauri E2E** — SP-8 (separate spec).
- **Multi-persona stress** — single-persona is the default; multi-persona is incidental. Add tests if/when a real multi-persona use case lands.
- **Long-soak (24h+)** — operational concern, not spec concern. Consider a nightly cron post-ship.
- **Performance benchmarks** — token-rate / time-to-first-byte measurements live in `benches/` if they ever land. Not required for SP-7 acceptance.

---

## 13. Out of Scope (Explicit)

These belong to other sub-projects or future versions of SP-7:

- **Multi-modal content** (images, audio in chat). Wire schema reserves room (`reply_chunk` could become `reply_chunk` + `reply_image` later) but no support today. → SP-8 / future.
- **TTS voice synthesis.** Not even an event slot today. → Future post-SP-8.
- **`GET /tools/log` endpoint.** The MCP server already writes the tool audit log (per `2026-04-27-mcp-config-path-for-brain-tools-design.md`). Live tool events flow through `WS /events` (`tool_call` + `tool_result`). No bridge endpoint for the historical log — read the file directly.
- **Auth / token-based access.** Localhost binding is the v1 security boundary. If the bridge ever exposes over a network (it shouldn't), bearer-token auth gets specced in SP-7.x.
- **Event replay / persistent event log.** Future SP-7.x.
- **Outbox / proactive messaging.** Event type reserved; no source emits it. → Future.
- **launchd/systemd integration.** Out of scope today; auto-spawn covers the CLI use case and Tauri owns the Tauri use case. If always-on becomes a goal, file a SP-7.x spec.
- **Recovery dream on dirty-shutdown.** OG's `recover_dirty_shutdown` runs a "recovery dream" + backup. SP-7's recovery only drains sessions; running a dream from a recovery path adds complexity (provider must be live, embeddings hot, soul accessible) that doesn't pay for itself today. Revisit if dirty shutdowns happen often enough to matter.

---

## 14. Implementation Order

For SP-7's writing-plans phase, the natural decomposition is six chunks. **Smoke test at every chunk boundary** — start the bridge, hit the new endpoint or behavior, watch the logs. Don't batch verification to the end.

| # | Chunk | Files touched | Smoke test |
|---|---|---|---|
| 1 | Event bus module + tests | `brain/bridge/events.py`, `tests/bridge/test_events.py` | `python -c "from brain.bridge.events import publish; publish('test')"` (no-op exits clean) |
| 2 | State file module (`bridge.json` schema, atomic write, dirty-shutdown predicate) + tests | `brain/bridge/daemon.py` (state-file portion), `tests/bridge/test_state_file.py` | `python -m pytest tests/bridge/test_state_file.py -v` |
| 3 | FastAPI server skeleton (lifespan, `/health`, `POST /session/new`, `GET /state/{sid}`) | `brain/bridge/server.py` | `nell bridge start --persona nell.sandbox && curl localhost:<port>/health \| jq` |
| 4 | Chat endpoints (`POST /chat`, `WS /stream` with simulated chunking + tool events) | `brain/bridge/server.py` (extend), `brain/bridge/chat_handlers.py` (if needed for clarity) | Real chat turn round-trip via WS |
| 5 | Supervisor thread + `/sessions/close` + idle-shutdown watcher | `brain/bridge/supervisor.py`, `brain/bridge/server.py` (extend) | Start bridge, send chat, wait 5+ min idle (or set `idle_threshold_min=0.1`), observe `session_closed` event + bridge shutdown |
| 6 | CLI integration (`nell bridge {start,stop,status,tail-events}`, `nell chat` auto-spawn-and-route) + dirty-shutdown recovery | `brain/cli.py`, `brain/bridge/daemon.py` (CLI handlers + recovery path) | Full flow: `nell bridge start`, `nell chat` (auto-routes), `nell bridge stop`. Then: kill -9 the daemon, `nell bridge start`, observe `recovered` event in `nell bridge tail-events` and drained sessions. |

Each chunk gets its own commit. Each commit is independently reviewable. The smoke-test column is the gate for moving to the next chunk.

---

## 15. References

- Master reference: `docs/superpowers/specs/2026-04-26-companion-emergence-master-reference.md` — §6 SP-7 scope, §7 Decision-Checking Guide, §8 Open Questions
- OG bridge ecosystem: `/Users/hanamori/NellBrain/nell_bridge.py`, `nell_bridge_session.py`, `nell_supervisor.py` — particularly `run_folded`, `recover_dirty_shutdown`, `EventBroadcaster`
- Vocabulary split (single-tenant guarantee): `2026-04-25-vocabulary-split-design.md:37, 460`
- Brain health module (walk + alarms surfaces): `2026-04-25-brain-health-module-design.md:144-204`
- Heartbeat engine (open/close trigger contract): `2026-04-23-week-4-heartbeat-engine-design.md:163-174`
- MCP brain-tools (tool-log coupling): `2026-04-27-mcp-config-path-for-brain-tools-design.md:130, 249`
- Principle alignment audit: `docs/superpowers/audits/2026-04-25-principle-alignment-audit.md`
- Pre-design audit: subagent punch list, this session 2026-04-28
