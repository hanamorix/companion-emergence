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
import logging
import os
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

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


async def _idle_watcher(state: "BridgeAppState", idle_shutdown_seconds: float) -> None:
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


def _respond_blocking(s: "BridgeAppState", sess: Any, message: str) -> Any:
    """Wrap brain.chat.engine.respond — blocks; called via asyncio.to_thread."""
    from brain.chat.engine import respond

    return respond(
        s.persona_dir,
        message,
        store=s.store,
        hebbian=s.hebbian,
        provider=s.provider,
        session=sess,
    )


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class NewSessionReq(BaseModel):
    client: str = "cli"  # "cli" | "tauri" | "tests"


class NewSessionResp(BaseModel):
    session_id: str
    persona: str
    created_at: str


class ChatReq(BaseModel):
    session_id: str
    message: str


class CloseReq(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# App state container — held on app.state.bridge
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
    supervisor_thread: Any | None = None  # Task 6 fills this


# ---------------------------------------------------------------------------
# Lifespan + app factory
# ---------------------------------------------------------------------------


def build_app(
    persona_dir: Path,
    client_origin: str = "cli",
    tick_interval_s: float = 60.0,
    silence_minutes: float = 5.0,
    idle_shutdown_seconds: float | None = None,
) -> FastAPI:
    """Build a FastAPI app for the given persona. Public for tests + daemon."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Construct singletons. Each opened DB handle is registered on an
        # ExitStack so a partial-init failure (e.g. HebbianMatrix raises
        # after MemoryStore opened) cleanly rolls back the handles already
        # acquired. Without this, a startup failure would leak fds.
        from contextlib import ExitStack

        cleanup = ExitStack()
        try:
            store = MemoryStore(persona_dir / "memories.db")
            cleanup.callback(store.close)
            hebbian = HebbianMatrix(persona_dir / "hebbian.db")
            cleanup.callback(hebbian.close)
            config = PersonaConfig.load(persona_dir / "persona_config.json")
            provider = get_provider(config.provider)

            embeddings = EmbeddingCache(
                persona_dir / "embeddings.db",
                FakeEmbeddingProvider(dim=256),
            )
            cleanup.callback(embeddings.close)
        except Exception:
            cleanup.close()
            raise

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

        # Spawn supervisor thread (non-daemon — joins on shutdown)
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
            # Cancel idle watcher first so it doesn't fire SIGTERM during teardown
            if idle_task is not None:
                idle_task.cancel()
                try:
                    await idle_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("idle watcher raised during teardown")
            # Stop supervisor thread
            stop_event.set()
            sup_thread.join(timeout=180.0)
            if sup_thread.is_alive():
                logger.warning("supervisor thread did not stop within 180s")
            # Drain all live sessions (silence_minutes=0) — late import so
            # monkeypatched close_stale_sessions in tests is honored.
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
            # ExitStack closes the three DB handles (registered LIFO):
            # embeddings → hebbian → store
            cleanup.close()
            logger.info("bridge stopped persona=%s", persona_dir.name)

    app = FastAPI(title="companion-emergence bridge", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        s: BridgeAppState = app.state.bridge
        uptime = (datetime.now(UTC) - s.started_at).total_seconds()

        # Walk + alarms — lightweight; defensive against fresh persona dirs
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
            "sessions_active": len(all_sessions()),
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

    # ── POST /chat — JSON one-shot fallback ────────────────────────────────
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
                result = await asyncio.to_thread(_respond_blocking, s, sess, req.message)
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

    # ── WS /stream/{session_id} — simulated streaming ──────────────────────
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

        chunk_delay_ms = int(os.environ.get("NELL_STREAM_CHUNK_DELAY_MS", "30"))

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

            # Tool events fire BEFORE reply chunks — they happened first in the loop.
            for inv in result.tool_invocations:
                tool_name = inv.get("name") or inv.get("tool_name") or "?"
                summary = inv.get("result_summary") or inv.get("summary") or ""
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
                    "at": _now(),
                }
            )
            events.publish(
                "chat_done",
                session_id=session_id,
                turn=result.turn,
                duration_ms=duration_ms,
            )

    # ── WS /events — server-push only broadcast ────────────────────────────
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

    # ── POST /sessions/close — explicit ingest trigger ─────────────────────
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
