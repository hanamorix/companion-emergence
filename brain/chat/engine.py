"""SP-6 chat engine — the keystone.

respond() is the single public entry point. It wires together:
  - voice.md (persona identity)
  - daemon state (residue from dream/heartbeat/reflex/research)
  - soul store (permanent crystallizations)
  - memory store (emotion state, memory search)
  - tool loop (provider.chat() + SP-3 dispatch)
  - ingest buffer (SP-4 persistence so chats become memories)

OG reference:
  - nell_bridge.py:run_tool_loop + _build_system_message + _persist_turn
  - nell_bridge_session.py:SessionState

Every error in the persistence path is caught and logged; it must never
break the response delivered to the user.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import LLMProvider
from brain.chat.prompt import build_system_message
from brain.chat.session import SessionState, create_session
from brain.chat.tool_loop import build_tools_list, run_tool_loop
from brain.chat.voice import load_voice
from brain.engines.daemon_state import load_daemon_state
from brain.ingest.buffer import ingest_turn
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """The outcome of one respond() call.

    Attributes
    ----------
    content:
        The assistant's reply text.
    session_id:
        The session this turn belongs to.
    turn:
        Turn index (1-based count after this turn was appended).
    tool_invocations:
        List of tool call records from the tool loop.
    duration_ms:
        Wall-clock time for the full respond() call in milliseconds.
    metadata:
        Catch-all for any extra info (provider name, etc.).
    """

    content: str
    session_id: str
    turn: int
    tool_invocations: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)


def respond(
    persona_dir: Path,
    user_input: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    session: SessionState | None = None,
    voice_md_override: str | None = None,
) -> ChatResult:
    """One chat turn end-to-end.

    Flow
    ----
    1. Resolve session (create if None)
    2. Load voice.md (or use override)
    3. Load DaemonState from persona_dir (SP-2)
    4. Open SoulStore from persona_dir (SP-5)
    5. Build system message via prompt.build_system_message
    6. Build messages list: [system, ...session.history, user]
    7. Run tool loop via tool_loop.run_tool_loop
    8. Persist turn (best-effort; errors logged + swallowed)
    9. Return ChatResult

    Parameters
    ----------
    persona_dir:
        Root directory for this persona's state files.
    user_input:
        The raw user message text for this turn.
    store:
        MemoryStore for emotion state aggregation + tool dispatch.
    hebbian:
        HebbianMatrix for tool dispatch.
    provider:
        LLMProvider to use for chat completion.
    session:
        Existing session to continue. Creates a new one if None.
    voice_md_override:
        If set, use this string instead of loading voice.md from disk.
        Useful for tests that don't want to write files.

    Returns
    -------
    ChatResult with content, session_id, turn count, tool invocations,
    and wall-clock duration.
    """
    t0 = time.monotonic()

    # 1. Session
    if session is None:
        session = create_session(persona_dir.name)

    # 2. Voice
    if voice_md_override is not None:
        voice_md = voice_md_override
    else:
        voice_md, voice_anomaly = load_voice(persona_dir)
        if voice_anomaly is not None:
            logger.warning(
                "voice.md anomaly: %s (%s) — continuing with recovered content",
                voice_anomaly.kind,
                voice_anomaly.action,
            )

    # 3. Daemon state
    daemon_state, daemon_anomaly = load_daemon_state(persona_dir)
    if daemon_anomaly is not None:
        logger.warning("daemon_state anomaly: %s", daemon_anomaly.kind)

    # 4. Soul store
    soul_db = persona_dir / "crystallizations.db"
    soul_store = SoulStore(str(soul_db))
    try:
        # 5. System message
        system_msg = build_system_message(
            persona_dir,
            voice_md=voice_md,
            daemon_state=daemon_state,
            soul_store=soul_store,
            store=store,
        )
    finally:
        soul_store.close()

    # 6. Messages list: [system, ...history, user]
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_msg),
        *session.history,
        ChatMessage(role="user", content=user_input),
    ]

    # 7. Tool loop
    tools = build_tools_list()
    response, invocations = run_tool_loop(
        messages,
        provider=provider,
        tools=tools,
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )
    content = response.content or ""

    # 8. Persist turn (best-effort)
    _persist_turn(
        persona_dir=persona_dir,
        session_id=session.session_id,
        user_text=user_input,
        assistant_text=content,
    )
    session.append_turn(user_input, content)

    duration_ms = int((time.monotonic() - t0) * 1000)

    return ChatResult(
        content=content,
        session_id=session.session_id,
        turn=session.turns,
        tool_invocations=invocations,
        duration_ms=duration_ms,
        metadata={"provider": provider.name()},
    )


def _persist_turn(
    persona_dir: Path,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Write both turns to the ingest buffer. Errors are logged, not raised.

    Mirrors OG _persist_turn (nell_bridge.py:200-230): failures here must
    never break the chat response. The ingest pipeline will pick up the
    buffer on session close (via nell chat REPL exit or supervisor sweep).
    """
    try:
        ingest_turn(persona_dir, {"session_id": session_id, "speaker": "user", "text": user_text})
        ingest_turn(
            persona_dir,
            {"session_id": session_id, "speaker": "assistant", "text": assistant_text},
        )
    except (OSError, ValueError) as exc:
        logger.warning(
            "conversation buffer write failed session=%s err=%s",
            session_id,
            exc,
        )
