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

from brain.bridge.chat import ChatMessage, ContentBlock, ImageBlock, TextBlock
from brain.bridge.provider import LLMProvider
from brain.chat.budget import apply_budget
from brain.chat.prompt import build_system_message
from brain.chat.session import SessionState, create_session
from brain.chat.tool_loop import build_tools_list, run_tool_loop
from brain.chat.voice import load_voice
from brain.engines.daemon_state import load_daemon_state
from brain.images import media_type_for_sha
from brain.ingest.buffer import ingest_turn, read_session
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
    image_shas: list[str] | None = None,
    reply_to_audit_id: str | None = None,
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
    reply_to_audit_id:
        If this turn is an explicit reply to an outbound initiate, the
        audit row id. Surfaced to build_system_message so the system
        prompt carries "you are replying to your earlier outbound about
        X" context. Bundle A #4 / v0.0.9 review TODO.

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
        # 5. System message — pass the user's current input so the recall
        # block can surface memories matching this turn (Phase 2.A).
        system_msg = build_system_message(
            persona_dir,
            voice_md=voice_md,
            daemon_state=daemon_state,
            soul_store=soul_store,
            store=store,
            user_input=user_input,
            reply_to_audit_id=reply_to_audit_id,
        )
    finally:
        soul_store.close()

    # 6. Messages list — buffer-driven, with budget guard.
    user_msg = _build_user_message(persona_dir, user_input, image_shas)
    try:
        prior_turns = read_session(persona_dir, session.session_id)
        # Note: on call N+1, _persist_turn has already written turn N to
        # the buffer (engine.py persists BEFORE session.append_turn), so
        # prior_turns reflects the complete prior history with no torn turn.
        history_msgs = _buffer_turns_to_messages(persona_dir, prior_turns)
    except Exception:
        logger.exception(
            "engine.respond: buffer read failed session=%s; falling back to session.history",
            session.session_id,
        )
        history_msgs = list(session.history)

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_msg),
        *history_msgs,
        user_msg,
    ]
    messages = apply_budget(
        messages,
        max_tokens=190_000,
        preserve_tail_msgs=40,
        provider=provider,
    )

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

    # 8. Persist turn (best-effort, but surfaced in metadata)
    persistence_ok, persistence_error = _persist_turn(
        persona_dir=persona_dir,
        session_id=session.session_id,
        user_text=user_input,
        assistant_text=content,
        image_shas=image_shas,
    )
    session.append_turn(user_input, content)

    duration_ms = int((time.monotonic() - t0) * 1000)

    return ChatResult(
        content=content,
        session_id=session.session_id,
        turn=session.turns,
        tool_invocations=invocations,
        duration_ms=duration_ms,
        metadata={
            "provider": provider.name(),
            "persistence_ok": persistence_ok,
            "persistence_error": persistence_error,
        },
    )


def _build_user_message(
    persona_dir: Path,
    user_input: str,
    image_shas: list[str] | None,
) -> ChatMessage:
    """Compose the per-turn user ChatMessage.

    Text-only turns produce the legacy str-content shape. Image-bearing
    turns produce a tuple of TextBlock + ImageBlocks; ImageBlock
    media_type is sniffed from disk via brain.images.media_type_for_sha.
    Missing or malformed images are logged and skipped — the chat must
    not break because an attachment vanished between upload and send.
    """
    if not image_shas:
        return ChatMessage(role="user", content=user_input)

    from brain.images import media_type_for_sha

    blocks: list[ContentBlock] = []
    if user_input:
        blocks.append(TextBlock(text=user_input))
    for sha in image_shas:
        try:
            media_type = media_type_for_sha(persona_dir, sha)
            blocks.append(ImageBlock(image_sha=sha, media_type=media_type))
        except (FileNotFoundError, ValueError) as exc:
            # Don't let a missing or malformed sha break the turn — log
            # and skip. The user's text portion still goes through.
            logger.warning(
                "skipping image_sha=%s: %s", sha[:8] if len(sha) >= 8 else sha, exc
            )
    if not blocks:
        # Defensive: every block dropped (unlikely). Fall back to text.
        return ChatMessage(role="user", content=user_input or "")
    return ChatMessage(role="user", content=tuple(blocks))


def _buffer_turns_to_messages(
    persona_dir: Path, turns: list[dict]
) -> list[ChatMessage]:
    """Reconstruct ChatMessage list from buffer JSONL records.

    Image-bearing user turns rebuild a (TextBlock, *ImageBlock) content
    tuple identical to what _build_user_message produces for live turns.
    Missing or unreadable images are skipped with a warning, matching
    _build_user_message's defensive behaviour.
    """
    out: list[ChatMessage] = []
    for t in turns:
        speaker = t.get("speaker")
        text = t.get("text", "")
        if speaker == "user":
            role = "user"
        elif speaker == "assistant":
            role = "assistant"
        else:
            continue  # skip unknown speakers defensively

        image_shas = t.get("image_shas") or []
        if role == "user" and image_shas:
            blocks: list[ContentBlock] = []
            if text:
                blocks.append(TextBlock(text=text))
            for sha in image_shas:
                try:
                    media_type = media_type_for_sha(persona_dir, sha)
                    blocks.append(ImageBlock(image_sha=sha, media_type=media_type))
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning(
                        "buffer replay: skipping image_sha=%s: %s",
                        sha[:8] if len(sha) >= 8 else sha,
                        exc,
                    )
            if blocks:
                out.append(ChatMessage(role=role, content=tuple(blocks)))
            elif text:
                out.append(ChatMessage(role=role, content=text))
            continue

        out.append(ChatMessage(role=role, content=text))
    return out


def _persist_turn(
    persona_dir: Path,
    session_id: str,
    user_text: str,
    assistant_text: str,
    image_shas: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Write both turns to the ingest buffer. Errors are logged, not raised.

    Mirrors OG _persist_turn (nell_bridge.py:200-230): failures here must
    never break the chat response. The ingest pipeline will pick up the
    buffer on session close (via nell chat REPL exit or supervisor sweep).
    """
    try:
        user_record: dict = {
            "session_id": session_id,
            "speaker": "user",
            "text": user_text,
        }
        if image_shas:
            user_record["image_shas"] = list(image_shas)
        ingest_turn(persona_dir, user_record)
        ingest_turn(
            persona_dir,
            {"session_id": session_id, "speaker": "assistant", "text": assistant_text},
        )
        return True, None
    except Exception as exc:  # noqa: BLE001
        # The contract here is explicit (per OG nell_bridge.py:200-230):
        # persistence errors must NEVER break the chat response. The chat
        # surface is more important than surfacing a buffer-layer bug to
        # the user. We log with full traceback so the bug isn't lost.
        logger.exception(
            "conversation buffer write failed session=%s",
            session_id,
        )
        return False, str(exc)
