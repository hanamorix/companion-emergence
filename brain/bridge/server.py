"""SP-7 FastAPI app — bridge daemon HTTP+WS server.

Exposes (Task 4):
  POST /session/new        — create a new session
  GET  /state/{session_id} — return session state
  GET  /health             — liveness + walk_persona + alarms

Chat endpoints added in Task 5:
  POST /chat               — JSON one-shot fallback
  WS   /stream/{sid}       — simulated word-by-word streaming
  WS   /events             — server-push broadcast
  POST /sessions/close     — explicit ingest trigger

Singletons are constructed once at lifespan startup and held on app.state.bridge:
  - MemoryStore, HebbianMatrix, EmbeddingCache, LLMProvider
  - EventBus
  - in_flight_locks: dict[session_id, asyncio.Lock]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator

from brain.bridge import events
from brain.bridge.events import EventBus
from brain.bridge.provider import LLMProvider, get_provider
from brain.chat.session import all_sessions, create_session, get_session
from brain.health.alarm import compute_pending_alarms
from brain.health.walker import walk_persona
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig

logger = logging.getLogger(__name__)

# Browser/WebView origins that are allowed to call the localhost bridge.
# HTTP routes are still bearer-token protected; CORS is only the browser's
# same-origin waiver. Keep this exact-origin list narrow so a random web page
# cannot probe bridge responses even if it can hit 127.0.0.1.
DEFAULT_ALLOWED_ORIGINS = (
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "null",
)
DEV_ALLOWED_ORIGINS = (
    # Tauri devUrl / Vite config for this app.
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    # Vite's defaults and adjacent fallback ports used by browser-mode docs.
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
)

# Audit 2026-05-07 P4-1: ChatReq.image_shas item-level validator. The
# ingest path keys cache lookups on these values, so a renderer compromise
# that posted "../../etc/passwd" would otherwise traverse the cache root.
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _word_chunks(text: str) -> list[str]:
    """Split text into word-or-whitespace tokens preserving spacing.

    'hello world  from' -> ['hello', ' ', 'world', '  ', 'from']
    Each chunk sent verbatim so reassembly == original text.
    """
    return re.findall(r"\S+|\s+", text)


async def _idle_watcher(state: BridgeAppState, idle_shutdown_seconds: float) -> None:
    """Background task that triggers graceful shutdown after idle threshold.

    Production-only path. Tests should NOT start this watcher (default
    idle_shutdown_seconds=None means no watcher) — firing SIGTERM in a test
    process would kill pytest.
    """
    while True:
        await asyncio.sleep(min(idle_shutdown_seconds, 60))
        if _check_idle(state, idle_shutdown_seconds):
            logger.info("idle shutdown firing — no traffic for >%ss", idle_shutdown_seconds)
            os.kill(os.getpid(), 15)  # SIGTERM, lifespan __aexit__ runs cleanup
            return


def _check_idle(state: Any, idle_shutdown_seconds: float) -> bool:
    """True if bridge should auto-shutdown.

    Pure predicate — no side effects. Conditions:
      - last activity (chat OR bridge startup) older than threshold
      - no active session has its in_flight lock held

    Bridge startup counts as activity so a freshly launched app has the
    full idle window before the watcher fires. Without that fallback,
    ``last_chat_at is None`` collapsed to "idle" and the bridge SIGTERM'd
    itself ~60s after every launch — which then triggered the
    close-heartbeat (decay + dream + reflex + growth) on every relaunch,
    looking from the UI like the brain was 'flooding'.
    """
    now = datetime.now(UTC)
    last_activity = state.last_chat_at or state.started_at
    if (now - last_activity).total_seconds() < idle_shutdown_seconds:
        return False
    for lock in state.in_flight_locks.values():
        if lock.locked():
            return False
    return True


def _respond_blocking(
    persona_dir: Path,
    sess: Any,
    message: str,
    provider: LLMProvider,
    image_shas: list[str] | None = None,
) -> Any:
    """Wrap brain.chat.engine.respond — blocks; called via asyncio.to_thread.

    Opens fresh per-call MemoryStore + HebbianMatrix INSIDE the worker thread,
    so SQLite connections never cross thread boundaries. Closing on exit means
    no leaked fds. Provider is passed in (stateless / thread-safe).
    """
    from contextlib import ExitStack

    from brain.chat.engine import respond

    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)
        return respond(
            persona_dir,
            message,
            store=store,
            hebbian=hebbian,
            provider=provider,
            session=sess,
            image_shas=image_shas,
        )


def _close_session_blocking(
    persona_dir: Path, session_id: str, provider: LLMProvider,
) -> Any:
    """Wrap brain.ingest.pipeline.close_session — blocks; called via asyncio.to_thread.

    Same per-call store pattern as _respond_blocking; close_session needs
    embeddings too for dedupe.
    """
    from contextlib import ExitStack

    from brain.ingest.pipeline import close_session

    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)
        embeddings = EmbeddingCache(
            persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256),
        )
        stack.callback(embeddings.close)
        return close_session(
            persona_dir,
            session_id,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )


async def _wait_for_in_flight_drain(state: BridgeAppState, *, timeout: float = 30.0) -> None:
    """Wait for all per-session in_flight locks to release, up to `timeout` seconds.

    Spec §7 step 2: graceful shutdown waits up to 30s for active chat turns
    to finish before proceeding to the close-and-stop steps. We poll every
    100ms — locks are asyncio.Lock so this stays on the loop.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if not any(lock.locked() for lock in state.in_flight_locks.values()):
            return
        await asyncio.sleep(0.1)
    held = sum(1 for lock in state.in_flight_locks.values() if lock.locked())
    if held:
        logger.warning("shutdown drain: %d in-flight chat(s) did not release in %.0fs",
                       held, timeout)


_CLOSE_HEARTBEAT_DEBOUNCE_S = 300.0


def _run_heartbeat_close(persona_dir: Path, provider: LLMProvider) -> None:
    """Fire HeartbeatEngine.run_tick(trigger='close') in-process.

    Per SP-7 spec §7 step 5 + Reflex Phase 2's anchor for weekly growth.
    Best-effort: any exception is logged by the caller; we don't block
    shutdown on heartbeat issues.

    Debounced: when a close-heartbeat fired within the last 5 minutes
    (e.g. during a dev cycle of repeated rebuild+relaunch), skip the
    decay/dream/reflex/growth tail and exit. The session-drain step
    above this call already saved everything; the tail is what causes
    the 'flooding' perception when the bridge restarts often.

    Per H-A: opens its own per-call stores inside the worker thread.
    Constructor pattern mirrors brain/cli.py:_heartbeat_handler.
    """
    from contextlib import ExitStack

    from brain.engines.heartbeat import HeartbeatEngine
    from brain.persona_config import PersonaConfig
    from brain.search.factory import get_searcher

    state_path = persona_dir / "heartbeat_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            last_run = state.get("last_close_at") or state.get("last_run")
            if last_run:
                if last_run.endswith("Z"):
                    last_run = last_run[:-1] + "+00:00"
                last_dt = datetime.fromisoformat(last_run)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=UTC)
                age = (datetime.now(UTC) - last_dt).total_seconds()
                if age < _CLOSE_HEARTBEAT_DEBOUNCE_S:
                    logger.info(
                        "close-heartbeat debounced (last fire %.0fs ago < %.0fs)",
                        age, _CLOSE_HEARTBEAT_DEBOUNCE_S,
                    )
                    return
        except Exception:  # noqa: BLE001
            logger.debug("close-heartbeat debounce check failed", exc_info=True)

    config = PersonaConfig.load(persona_dir / "persona_config.json")
    searcher = get_searcher(config.searcher)
    default_arcs_path = (
        Path(__file__).parent.parent / "engines" / "default_reflex_arcs.json"
    )

    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)

        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=provider,
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            reflex_arcs_path=persona_dir / "reflex_arcs.json",
            reflex_log_path=persona_dir / "reflex_log.json",
            reflex_default_arcs_path=default_arcs_path,
            searcher=searcher,
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=Path(__file__).parent.parent
            / "engines" / "default_interests.json",
            persona_name=persona_dir.name,
            persona_system_prompt=f"You are {persona_dir.name}.",
        )
        engine.run_tick(trigger="close", dry_run=False)


def _drain_sessions_blocking(
    persona_dir: Path, provider: LLMProvider, silence_minutes: float = 0,
) -> Any:
    """Wrap brain.ingest.pipeline.close_stale_sessions — used by shutdown.

    Same per-call store pattern. Silence_minutes=0 (default) closes EVERY
    live session, which is what graceful shutdown wants.
    """
    from contextlib import ExitStack

    from brain.ingest.pipeline import close_stale_sessions

    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)
        embeddings = EmbeddingCache(
            persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256),
        )
        stack.callback(embeddings.close)
        return close_stale_sessions(
            persona_dir,
            silence_minutes=silence_minutes,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class NewSessionReq(BaseModel):
    client: Literal["cli", "tauri", "tests"] = "cli"


class NewSessionResp(BaseModel):
    session_id: str
    persona: str
    created_at: str


class ChatReq(BaseModel):
    session_id: str = Field(..., min_length=36, max_length=36, pattern=r"^[0-9a-fA-F-]{36}$")
    message: str = Field(..., min_length=1, max_length=20_000)
    # Optional image attachments — sha-strings as returned by /upload.
    # Audit 2026-05-07 P4-1: comment used to promise 64-char-hex
    # validation but the model only enforced the list length cap.
    # Now Pydantic enforces it at the API boundary too. Deeper image
    # handling stays as defense in depth.
    image_shas: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    @field_validator("image_shas")
    @classmethod
    def _validate_image_shas(cls, v: list[str]) -> list[str]:
        for sha in v:
            if not _SHA256_HEX_RE.fullmatch(sha):
                raise ValueError(
                    f"image_sha must be 64 lowercase hex chars, got {sha!r}"
                )
        return v


class CloseReq(BaseModel):
    session_id: str = Field(..., min_length=36, max_length=36, pattern=r"^[0-9a-fA-F-]{36}$")


# ---------------------------------------------------------------------------
# App state container — held on app.state.bridge
# ---------------------------------------------------------------------------


@dataclass
class BridgeAppState:
    """Bridge runtime state held on app.state.bridge.

    Note: SQLite-backed stores (MemoryStore, HebbianMatrix, EmbeddingCache)
    are NOT held here. Each worker thread / handler that needs them opens
    its own per-call instances against `persona_dir`. The `provider` is
    safe to share — it's stateless (Claude CLI invokes a subprocess per
    call; no long-lived resource).

    `auth_token` (H-C): None disables auth (test/dev). When set, all HTTP
    routes require Authorization: Bearer <token>; WS endpoints prefer
    Sec-WebSocket-Protocol: bearer, <token>, plus Origin allowlist.
    """

    persona_dir: Path
    persona: str
    client_origin: str
    started_at: datetime
    provider: LLMProvider
    event_bus: EventBus
    in_flight_locks: dict[str, asyncio.Lock]
    last_chat_at: datetime | None = None
    supervisor_thread: Any | None = None
    auth_token: str | None = None


# ---------------------------------------------------------------------------
# Lifespan + app factory
# ---------------------------------------------------------------------------


def build_app(
    persona_dir: Path,
    client_origin: str = "cli",
    tick_interval_s: float = 60.0,
    silence_minutes: float = 5.0,
    idle_shutdown_seconds: float | None = None,
    auth_token: str | None = None,
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS,
) -> FastAPI:
    """Build a FastAPI app for the given persona. Public for tests + daemon.

    auth_token: when set, HTTP routes require Authorization: Bearer <token>
    and WS endpoints require Sec-WebSocket-Protocol: bearer, <token>.
    None (default) disables auth — used by tests and offline dev. Production
    runner.py always passes a fresh ephemeral token.

    allowed_origins: WebSocket Origin header allowlist (extra defense
    against browser-based attacks if someone proxies localhost). "null"
    matches CLI/non-browser clients; "tauri://localhost" matches SP-8.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Per HA hardening: NO persistent SQLite stores held on app.state.
        # Each worker thread / handler opens its own per-call stores against
        # persona_dir. The lifespan only constructs the provider (stateless),
        # the EventBus, the supervisor thread, and the idle watcher.
        config = PersonaConfig.load(persona_dir / "persona_config.json")
        provider = get_provider(config.provider)

        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())
        events.set_publisher(bus.publish)

        app.state.bridge = BridgeAppState(
            persona_dir=persona_dir,
            persona=persona_dir.name,
            client_origin=client_origin,
            started_at=datetime.now(UTC),
            provider=provider,
            event_bus=bus,
            in_flight_locks={},
            auth_token=auth_token,
        )
        logger.info("bridge started persona=%s pid=%d", persona_dir.name, os.getpid())

        # Spawn supervisor thread (non-daemon — joins on shutdown)
        from brain.bridge.supervisor import run_folded

        stop_event = threading.Event()
        sup_thread = threading.Thread(
            target=run_folded,
            kwargs={
                "stop_event": stop_event,
                "persona_dir": persona_dir,
                "provider": provider,
                "event_bus": bus,
                "tick_interval_s": tick_interval_s,
                "silence_minutes": silence_minutes,
            },
            name="sp7-supervisor",
            daemon=False,
        )
        sup_thread.start()
        app.state.bridge.supervisor_thread = sup_thread

        # Idle-shutdown watcher (only if requested)
        idle_task = None
        if idle_shutdown_seconds is not None and idle_shutdown_seconds > 0:
            idle_task = asyncio.create_task(
                _idle_watcher(app.state.bridge, idle_shutdown_seconds)
            )

        try:
            yield
        finally:
            # Shutdown sequence per spec §7:
            #   1. Cancel idle watcher (so it can't fire SIGTERM during teardown)
            #   2. Drain in-flight chats (best-effort wait, 30s cap)
            #   3. Close all live sessions via ingest pipeline (silence_minutes=0)
            #   4. Stop supervisor thread (180s join cap)
            #   5. Heartbeat close-trigger (Reflex Phase 2 growth tick anchor)
            #   6. Publish shutdown event
            #   7. Clear publisher

            # 1. Cancel idle watcher
            if idle_task is not None:
                idle_task.cancel()
                try:
                    await idle_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("idle watcher raised during teardown")

            # 2. Drain in-flight chats — wait up to 30s for active locks to release
            await _wait_for_in_flight_drain(app.state.bridge, timeout=30.0)

            # 3. Close all live sessions (silence_minutes=0) via per-call stores.
            #    This is the data-saving step — every conversation that was open
            #    becomes memory before the lights go out.
            try:
                reports = await asyncio.to_thread(
                    _drain_sessions_blocking, persona_dir, provider, 0,
                )
            except Exception:
                logger.exception("shutdown drain failed")
                reports = []

            # 3b. Record drain-error count to the state file. The runner's
            # `_write_clean_shutdown` (atexit + finally) reads this and
            # leaves shutdown_clean=False if any session failed to ingest,
            # so the next bridge start re-runs `run_recovery_if_needed`
            # against the orphan buffers instead of treating the dirty
            # exit as clean. This closes the "clean shutdown that
            # happened to have a failed drain" hole.
            drain_errors = sum(getattr(r, "errors", 0) for r in reports)
            if drain_errors > 0:
                try:
                    from brain.bridge import state_file as _state_file_mod
                    cur = _state_file_mod.read(persona_dir)
                    if cur is not None:
                        cur.drain_errors = drain_errors
                        _state_file_mod.write(persona_dir, cur)
                    logger.warning(
                        "shutdown drain produced %d ingest errors; marked dirty for next start",
                        drain_errors,
                    )
                except Exception:
                    logger.exception("failed to record drain_errors to state file")

            # 4. Stop supervisor thread
            stop_event.set()
            sup_thread.join(timeout=180.0)
            if sup_thread.is_alive():
                logger.warning("supervisor thread did not stop within 180s")

            # 5. Heartbeat close-trigger — anchor for Reflex Phase 2 weekly growth.
            #    In-process import + run_tick(trigger="close"). Best-effort:
            #    failure is logged but doesn't block shutdown.
            try:
                await asyncio.to_thread(_run_heartbeat_close, persona_dir, provider)
            except Exception:
                logger.exception("heartbeat close-trigger failed during shutdown")

            # 6. Publish shutdown event
            try:
                bus.publish(
                    {"type": "shutdown", "clean": True, "drained": len(reports), "at": _now()}
                )
            except Exception:
                logger.exception("shutdown event publish failed")

            # 7. Clear publisher
            events.set_publisher(None)
            logger.info("bridge stopped persona=%s", persona_dir.name)

    app = FastAPI(title="companion-emergence bridge", version="0.1.0", lifespan=lifespan)

    # CORS — narrowly scoped to the same allowed_origins used for WS auth,
    # plus localhost dev origins (Tauri devUrl + Vite default ports). Bearer
    # auth still gates every route — CORS is not a security boundary, just a
    # same-origin policy waiver for the trusted local app surface.
    from fastapi.middleware.cors import CORSMiddleware

    cors_origins = list(allowed_origins) + list(DEV_ALLOWED_ORIGINS)
    # Extend the WS Origin allowlist with the same dev origins so
    # browser-mode WebSocket connections to /stream pass the Origin
    # check. Bearer subprotocol auth still gates every connection.
    allowed_origins = tuple(list(allowed_origins) + list(DEV_ALLOWED_ORIGINS))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── H-C: auth + Origin check helpers ──────────────────────────────────
    # HTTP: require `Authorization: Bearer <token>` when auth_token is set.
    # WS: require `Sec-WebSocket-Protocol: bearer, <token>` + Origin allowlist.
    # Both no-op when auth_token is None (test/dev mode).

    import secrets as _secrets

    from fastapi import Header

    def _consteq(a: str, b: str) -> bool:
        """Constant-time string compare. secrets.compare_digest handles
        tokens of different lengths safely."""
        return _secrets.compare_digest(a.encode(), b.encode())

    # auth_token captured by closure; tests pass None to disable.
    async def require_http_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        if auth_token is None:
            return  # auth disabled (tests / offline dev)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization[len("Bearer "):]
        if not _consteq(token, auth_token):
            raise HTTPException(status_code=401, detail="invalid token")

    def _ws_subprotocol_parts(ws: WebSocket) -> list[str]:
        raw = ws.headers.get("sec-websocket-protocol", "")
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _ws_subprotocol_token(ws: WebSocket) -> str:
        """Extract a browser-friendly WS bearer token from subprotocols.

        Supported form: Sec-WebSocket-Protocol: bearer, <token>.
        """
        parts = _ws_subprotocol_parts(ws)
        for i, part in enumerate(parts[:-1]):
            if part.lower() == "bearer":
                return parts[i + 1]
        return ""

    def _ws_accept_subprotocol(ws: WebSocket) -> str | None:
        parts = _ws_subprotocol_parts(ws)
        return "bearer" if any(part.lower() == "bearer" for part in parts) else None

    def _check_ws_auth(ws: WebSocket) -> tuple[bool, str]:
        """Return (ok, reason). Caller closes the WS with reason on False."""
        # Origin check first — cheap, defends against browsers.
        origin = ws.headers.get("origin") or "null"
        if origin not in allowed_origins:
            logger.warning("WS rejected: origin=%r not in allowlist", origin)
            return False, "origin not allowed"
        # Token check second (closure-captured auth_token).
        if auth_token is None:
            return True, ""  # auth disabled
        token = _ws_subprotocol_token(ws)
        if not token:
            return False, "missing token"
        if not _consteq(token, auth_token):
            return False, "invalid token"
        return True, ""

    @app.get("/health", dependencies=[Depends(require_http_auth)])
    def health() -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        uptime = (datetime.now(UTC) - s.started_at).total_seconds()

        # Walk + alarms — lightweight; defensive against fresh persona dirs.
        # Narrow tuple so programming bugs (KeyError, AttributeError) inside
        # walk_persona / compute_pending_alarms surface as 500 rather than
        # leaving /health silently green forever.
        health_scan = "ok"
        health_error: str | None = None
        try:
            anomalies = walk_persona(s.persona_dir)
            alarms = compute_pending_alarms(s.persona_dir)
        except (OSError, sqlite3.Error, ValueError) as exc:
            logger.warning("health walk failed", exc_info=True)
            anomalies = []
            alarms = []
            health_scan = "failed"
            health_error = f"{type(exc).__name__}: {exc}"

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
            "sessions_active": len(all_sessions()),
            "last_chat_at": s.last_chat_at.isoformat() if s.last_chat_at else None,
            "supervisor_thread": sup_status,
            "health_scan": health_scan,
            "health_error": health_error,
            "pending_alarms": len(alarms),
            "anomalies": len(anomalies),
        }

    @app.post("/session/new", response_model=NewSessionResp, dependencies=[Depends(require_http_auth)])
    def session_new(req: NewSessionReq) -> NewSessionResp:
        s: BridgeAppState = app.state.bridge
        sess = create_session(s.persona)
        return NewSessionResp(
            session_id=sess.session_id,
            persona=s.persona,
            created_at=sess.created_at.isoformat(),
        )

    @app.get("/state/{session_id}", dependencies=[Depends(require_http_auth)])
    def state_endpoint(session_id: str) -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        sess = get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="session not found")
        in_flight = (
            session_id in s.in_flight_locks
            and s.in_flight_locks[session_id].locked()
        )
        return {
            "session_id": sess.session_id,
            "persona": sess.persona_name,
            "turns": sess.turns,
            "last_turn_at": sess.last_turn_at.isoformat() if sess.last_turn_at else None,
            "history_len": len(sess.history),
            "in_flight": in_flight,
        }

    # ── /persona/state — NellFace app panels' aggregated read ──────────────
    @app.get("/persona/state", dependencies=[Depends(require_http_auth)])
    def get_persona_state() -> dict[str, Any]:
        """Aggregated persona state for the NellFace UI panels.

        Composes emotions / body / interior / soul_highlight / mode in a
        single round-trip. Fail-soft per subsystem — fresh personas or
        partial data still return 200 with the available pieces.
        """
        from brain.bridge.persona_state import build_persona_state
        return build_persona_state(persona_dir)

    # ── /self/works[*] — self-knowledge surface (source spec §15.2) ────────
    @app.get("/self/works", dependencies=[Depends(require_http_auth)])
    def get_self_works(type: str | None = None, limit: int = 20) -> dict:
        """List recent works (most recent first). Optional ?type=NAME."""
        from brain.tools.impls.list_works import list_works
        return {"works": list_works(type=type, limit=limit, persona_dir=persona_dir)}

    @app.get("/self/works/search", dependencies=[Depends(require_http_auth)])
    def search_self_works(q: str, type: str | None = None, limit: int = 20) -> dict:
        """Full-text search over works. ?q=QUERY required."""
        from brain.tools.impls.search_works import search_works
        return {"works": search_works(query=q, type=type, limit=limit, persona_dir=persona_dir)}

    @app.get("/self/works/{work_id}", dependencies=[Depends(require_http_auth)])
    def get_self_work_by_id(work_id: str) -> dict:
        """Return one work's full content + metadata."""
        from brain.tools.impls.read_work import read_work
        result = read_work(id=work_id, persona_dir=persona_dir)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    # ── POST /upload — multimodal image upload ────────────────────────────
    # Image upload limit (matches the spec D2 default; per-persona override
    # via PersonaConfig.image_max_bytes can come later if needed).
    _IMAGE_MAX_BYTES = 20 * 1024 * 1024  # noqa: N806 — local frozen constant
    _ALLOWED_UPLOAD_MEDIA_TYPES = frozenset(  # noqa: N806
        {"image/png", "image/jpeg", "image/webp", "image/gif"}
    )

    @app.post("/upload", dependencies=[Depends(require_http_auth)])
    async def upload(file: UploadFile) -> dict[str, Any]:
        """Accept a multipart-uploaded image, persist content-addressably.

        Returns ``{sha, media_type, size_bytes}`` on success. Image lands at
        ``<persona_dir>/images/<sha>.<ext>``. Identical content (same sha)
        is deduped — second upload of the same bytes returns the same sha
        without writing a duplicate file.

        Errors:
          * 415 — unsupported media_type (only PNG / JPEG / WebP / GIF)
          * 413 — file exceeds the 20 MB cap
        """
        from brain.images import save_image_bytes, sniff_media_type

        s: BridgeAppState = app.state.bridge
        declared_media_type = (file.content_type or "").lower()
        if declared_media_type not in _ALLOWED_UPLOAD_MEDIA_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported media_type {declared_media_type!r}; "
                f"must be one of {sorted(_ALLOWED_UPLOAD_MEDIA_TYPES)}",
            )
        # Read up to limit + 1 so we can detect overrun without buffering
        # arbitrarily large payloads in memory.
        raw = await file.read(_IMAGE_MAX_BYTES + 1)
        if len(raw) > _IMAGE_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file too large; max {_IMAGE_MAX_BYTES} bytes",
            )
        # Sniff the actual bytes — the multipart Content-Type header is
        # client-controlled and a renderer compromise could ship arbitrary
        # bytes under an image MIME label. The disk's ground truth is
        # what gets passed to the provider later, so this is the gate.
        sniffed = sniff_media_type(raw)
        if sniffed is None:
            raise HTTPException(
                status_code=422,
                detail="image bytes don't match any supported format",
            )
        if sniffed != declared_media_type:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"declared {declared_media_type!r} but bytes look like "
                    f"{sniffed!r}"
                ),
            )
        record = save_image_bytes(s.persona_dir, raw, sniffed)
        return {
            "sha": record.sha,
            "media_type": record.media_type,
            "size_bytes": record.size_bytes,
        }

    # ── POST /chat — JSON one-shot fallback ────────────────────────────────
    @app.post("/chat", dependencies=[Depends(require_http_auth)])
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
                    _respond_blocking,
                    s.persona_dir,
                    sess,
                    req.message,
                    s.provider,
                    req.image_shas or None,
                )
            except Exception as exc:
                logger.exception("chat failed session=%s", req.session_id)
                # Audit 2026-05-07 P3-2: keep detailed exception text in
                # logs only — clients get a stable code, not stderr or
                # local paths from the underlying provider/process.
                raise HTTPException(
                    status_code=502, detail="provider_failed"
                ) from exc
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
                "metadata": result.metadata,
            }

    # ── WS /stream/{session_id} — simulated streaming ──────────────────────
    @app.websocket("/stream/{session_id}")
    async def stream(ws: WebSocket, session_id: str) -> None:
        # H-C: validate token + Origin BEFORE accepting the upgrade.
        ok, reason = _check_ws_auth(ws)
        if not ok:
            await ws.close(code=4001, reason=reason)
            return
        await ws.accept(subprotocol=_ws_accept_subprotocol(ws))
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

        # Audit 2026-05-07 P3-1: receive raw text + bound the frame
        # before json.loads. The local + authenticated bridge isn't
        # internet-exposed, but a compromised renderer could still
        # cause a memory spike by sending a huge JSON frame; reject
        # at the byte boundary so the parser never sees oversized
        # payloads. 64 KB is generous against a 20k-char message
        # plus image_shas (each ≤64 hex × ≤8 = ~520 bytes) plus
        # JSON overhead.
        _WS_FRAME_MAX_BYTES = 64 * 1024  # noqa: N806 — local constant
        try:
            raw_frame = await ws.receive_text()
        except (WebSocketDisconnect, ValueError):
            return
        if len(raw_frame.encode("utf-8")) > _WS_FRAME_MAX_BYTES:
            await ws.send_json(
                {"type": "error", "code": "frame_too_large", "done": True}
            )
            await ws.close()
            return
        try:
            req = json.loads(raw_frame)
        except (ValueError, json.JSONDecodeError):
            await ws.send_json(
                {"type": "error", "code": "invalid_json", "done": True}
            )
            await ws.close()
            return
        if not isinstance(req, dict):
            await ws.send_json(
                {"type": "error", "code": "invalid_frame_shape", "done": True}
            )
            await ws.close()
            return
        message = req.get("message", "")
        if not isinstance(message, str) or not message:
            await ws.send_json({"type": "error", "code": "empty_message", "done": True})
            await ws.close()
            return
        if len(message) > 20_000:
            await ws.send_json({"type": "error", "code": "message_too_large", "done": True})
            await ws.close()
            return

        # Optional image attachments — sha-strings as returned by /upload.
        # Audit 2026-05-07 P4-1: same item-level constraints as the
        # HTTP /chat ChatReq path — list of strings only, ≤ 8 entries,
        # each entry must be 64 lowercase hex chars.
        raw_shas = req.get("image_shas") or []
        image_shas: list[str] | None = None
        if isinstance(raw_shas, list) and raw_shas:
            valid = (
                len(raw_shas) <= 8
                and all(isinstance(x, str) for x in raw_shas)
                and all(_SHA256_HEX_RE.fullmatch(x) for x in raw_shas)
            )
            if not valid:
                await ws.send_json(
                    {"type": "error", "code": "invalid_image_shas", "done": True}
                )
                await ws.close()
                return
            image_shas = list(raw_shas)

        chunk_delay_ms = int(os.environ.get("NELL_STREAM_CHUNK_DELAY_MS", "30"))

        async with lock:
            t0 = datetime.now(UTC)
            await ws.send_json({"type": "started", "session_id": session_id, "at": _now()})
            events.publish("chat_started", session_id=session_id, client=s.client_origin)

            try:
                result = await asyncio.to_thread(
                    _respond_blocking,
                    s.persona_dir,
                    sess,
                    message,
                    s.provider,
                    image_shas,
                )
            except Exception:
                logger.exception("stream failed session=%s", session_id)
                # Audit 2026-05-07 P3-2: stable code for clients;
                # full exception text stays in the log only.
                await ws.send_json(
                    {"type": "error", "code": "provider_failed", "done": True}
                )
                await ws.close()
                return

            # Tool events fire BEFORE reply chunks — they happened first in the loop.
            # tool_invocations shape per brain/chat/tool_loop.py:79 —
            # {name, arguments, result_summary, error?}. Pinned to canonical keys.
            for inv in result.tool_invocations:
                tool_name = inv.get("name", "?")
                summary = inv.get("result_summary", "")
                await ws.send_json(
                    {"type": "tool_call", "tool": tool_name, "session_id": session_id, "at": _now()}
                )
                await ws.send_json(
                    {
                        "type": "tool_result",
                        "tool": tool_name,
                        "summary": summary,
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
                    "metadata": result.metadata,
                    "at": _now(),
                }
            )
            events.publish(
                "chat_done",
                session_id=session_id,
                turn=result.turn,
                duration_ms=duration_ms,
            )
            # Explicit clean close — without this, FastAPI/Starlette
            # tears down the WS when the handler returns and the
            # browser sees an abnormal closure (1006). Sending
            # code=1000 completes the WS close handshake so the
            # client gets the clean shutdown it expects after the
            # `done` frame.
            await ws.close(code=1000)

    # ── WS /events — server-push only broadcast ────────────────────────────
    @app.websocket("/events")
    async def events_ws(ws: WebSocket) -> None:
        # H-C: validate token + Origin BEFORE accepting the upgrade.
        ok, reason = _check_ws_auth(ws)
        if not ok:
            await ws.close(code=4001, reason=reason)
            return
        await ws.accept(subprotocol=_ws_accept_subprotocol(ws))
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

    # ── POST /sessions/close — explicit ingest trigger ─────────────────────
    @app.post("/sessions/close", dependencies=[Depends(require_http_auth)])
    async def sessions_close(req: CloseReq) -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        from brain.chat.session import get_session, remove_session

        # H2/D2: differentiate cases.
        # Unknown session id (never registered, or already removed) → 404.
        # Known session → run ingest, remove from registry, drop in_flight lock.
        sess = get_session(req.session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="session not found")

        # Audit 2026-05-07 P2-4: serialise close behind the same per-
        # session lock /chat and /stream use. Without this, a renderer
        # close (e.g. Cmd-Q during a streaming reply) could close the
        # session mid-turn — ingesting a partial buffer, removing the
        # in-memory session, and dropping the lock while the chat
        # worker still thought the session was active. Acquiring the
        # lock here means close waits for any in-flight turn to
        # finish before running ingest.
        lock = s.in_flight_locks.setdefault(req.session_id, asyncio.Lock())
        async with lock:
            # H-A: per-call stores inside the worker thread, not shared singletons.
            # Wrap the pipeline call so internal exceptions don't crash the handler.
            try:
                report = await asyncio.to_thread(
                    _close_session_blocking, s.persona_dir, req.session_id, s.provider,
                )
            except Exception as exc:
                logger.exception("close_session failed session=%s", req.session_id)
                events.publish(
                    "session_close_failed",
                    session_id=req.session_id,
                    committed=0,
                    deduped=0,
                    soul_candidates=0,
                    soul_queue_errors=0,
                    errors=1,
                )
                # Keep the in-memory session and lock entry registered. The
                # buffer is still on disk when close_session raises, and a
                # caller or supervisor retry should be able to close the same
                # session id instead of seeing a false "closed" success.
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "ingest_failed",
                        "session_id": req.session_id,
                        "closed": False,
                        "committed": 0,
                        "deduped": 0,
                        "soul_candidates": 0,
                        "soul_queue_errors": 0,
                        "errors": 1,
                    },
                ) from exc

            if report.errors > 0:
                events.publish(
                    "session_close_failed",
                    session_id=req.session_id,
                    committed=report.committed,
                    deduped=report.deduped,
                    soul_candidates=report.soul_candidates,
                    soul_queue_errors=report.soul_queue_errors,
                    errors=report.errors,
                )
                # Extraction failures deliberately retain the JSONL buffer for
                # retry. Do not remove the in-memory session or drop its lock
                # entry, otherwise the retry path turns into 404 while the
                # client was told nothing actionable.
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "ingest_failed",
                        "session_id": req.session_id,
                        "closed": False,
                        "committed": report.committed,
                        "deduped": report.deduped,
                        "soul_candidates": report.soul_candidates,
                        "soul_queue_errors": report.soul_queue_errors,
                        "errors": report.errors,
                    },
                )

            # H2: clean up registry + lock so sessions_active stays accurate.
            remove_session(req.session_id)
        s.in_flight_locks.pop(req.session_id, None)

        events.publish(
            "session_closed",
            session_id=req.session_id,
            committed=report.committed,
            deduped=report.deduped,
            soul_candidates=report.soul_candidates,
            soul_queue_errors=report.soul_queue_errors,
            errors=report.errors,
        )
        return {
            "session_id": req.session_id,
            "closed": True,
            "committed": report.committed,
            "deduped": report.deduped,
            "soul_candidates": report.soul_candidates,
            "soul_queue_errors": report.soul_queue_errors,
            "errors": report.errors,
        }

    return app
