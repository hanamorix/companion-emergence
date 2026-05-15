"""LLM provider abstraction — ABC + concrete providers + factory.

Two call shapes coexist here:

  generate(prompt, *, system) -> str
      Simple single-turn text generation.  All existing engines (dream,
      heartbeat, reflex, research, growth) use this surface.  Do not change it.

  chat(messages, *, tools, options) -> ChatResponse
      Multi-turn structured chat with optional tool-calling.  Used by the
      chat engine, brain-tools, and the bridge daemon (SP-3, SP-4, SP-6, SP-7).

Both shapes live on LLMProvider so callers can swap providers without caring
which shape they need.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from brain.bridge.chat import (
    ChatMessage,
    ChatResponse,
    ImageBlock,
    TextBlock,
    ToolCall,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 300
_PROVIDER_CONTEXT_OPTION_KEYS = frozenset({"persona_dir"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Raised when a provider call fails with structured context.

    Attributes
    ----------
    stage:
        Short machine-readable tag identifying where the failure occurred
        (e.g. "ollama_http", "ollama_parse", "claude_cli_timeout").
    detail:
        Human-readable explanation with whatever context is available.
    """

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"[{stage}] {detail}")
        self.stage = stage
        self.detail = detail


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract LLM provider.  Subclasses implement both call shapes."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the LLM's completion for the given prompt.

        Unchanged from Week-1 contract.  Engines (dream, heartbeat, reflex,
        research, growth) call this — do not touch it.
        """

    def complete(self, prompt: str) -> str:
        """Compatibility shim — the initiate pipeline calls ``.complete()``.

        Delegates to :meth:`generate` with no system prompt. Concrete
        providers do not need to override this; the default implementation
        works for every provider that satisfies the ``generate`` contract.
        Engines that need a system prompt should call ``generate`` directly.
        """
        return self.generate(prompt, system=None)

    @abstractmethod
    def name(self) -> str:
        """Return a short provider name (e.g. 'fake', 'claude-cli:sonnet')."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Multi-turn structured chat with optional tool-calls.

        Parameters
        ----------
        messages:
            Conversation history in chronological order.
        tools:
            Optional list of tool schemas (raw JSON-schema dicts in the shape
            OpenAI / Ollama / Claude all consume).  None means no tool-calling.
        options:
            Provider-specific generation options (temperature, top_p, etc.).
        """

    def healthy(self) -> bool:
        """Quick liveness check.  Default: assume healthy.

        Providers that can check (e.g. Ollama /api/tags) override this.
        ClaudeCliProvider and FakeProvider default to True.
        """
        return True


# ---------------------------------------------------------------------------
# FakeProvider — deterministic, zero network
# ---------------------------------------------------------------------------


class FakeProvider(LLMProvider):
    """Deterministic hash-based echo provider for tests — zero network calls."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        seed_input = (system or "").encode("utf-8") + b"||" + prompt.encode("utf-8")
        h = hashlib.sha256(seed_input).hexdigest()[:16]
        return f"DREAM: test dream {h} — an associative thread"

    def name(self) -> str:
        return "fake"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Return a deterministic ChatResponse.

        tool_calls is always empty — FakeProvider never synthesises tool calls.
        The content hash is derived from the serialised messages so tests can
        assert determinism without caring about the exact value.
        """
        seed = json.dumps([m.to_dict() for m in messages], sort_keys=True).encode()
        h = hashlib.sha256(seed).hexdigest()[:16]
        return ChatResponse(
            content=f"FAKE_CHAT: response {h}",
            tool_calls=(),
            raw=None,
        )


# ---------------------------------------------------------------------------
# ClaudeCliProvider — shells out to `claude` CLI
# ---------------------------------------------------------------------------


def _claude_failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Return a useful error detail for failed Claude CLI subprocesses.

    Claude Code sometimes exits non-zero with an empty stderr but a structured
    JSON stdout payload, for example subscription quota responses:
    ``{"is_error": true, "api_error_status": 429, "result": "..."}``.
    If we only log stderr, background heartbeat failures become opaque
    ``exit 1:`` lines. Prefer the structured result when present, then fall
    back to stderr/stdout snippets.
    """
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            parts: list[str] = []
            if payload.get("api_error_status") is not None:
                parts.append(f"api_error_status={payload['api_error_status']}")
            if payload.get("is_error") is not None:
                parts.append(f"is_error={payload['is_error']}")
            if payload.get("result"):
                parts.append(str(payload["result"]))
            if parts:
                if stderr:
                    parts.append(f"stderr={stderr}")
                return "; ".join(parts)
        if not stderr:
            return f"stdout={stdout[:500]}"
    return stderr


class ClaudeCliProvider(LLMProvider):
    """Shells out to `claude -p <prompt> --output-format json`.

    Uses Hana's Claude Code subscription — no per-token API billing.
    Per the feedback memory: this is the default Claude path for
    companion-emergence and Hana's other projects.

    Chat surface note
    -----------------
    The Claude CLI (as of 2026-04) exposes only text-based input:
      - ``-p / --print`` for the user turn
      - ``--system-prompt`` for the system message
      - ``--input-format`` supports "text" (default) or "stream-json" but NOT
        a structured messages-array format.

    Therefore ClaudeCliProvider.chat() flattens the messages array into the
    text surface:
      - The first "system" message → ``--system-prompt``.
      - A single remaining user message is passed verbatim via ``-p``.
      - Multi-turn history is serialised as JSONL context data, not a
        ``User:`` / ``Assistant:`` transcript. Those canonical labels make
        Claude more likely to continue the transcript and leak labels into
        Nell's reply.

    Tool-calling via --mcp-config
    -----------------------------
    When ``tools`` is provided, the provider writes a temp mcp.json pointing
    at ``brain.mcp_server`` and passes ``--mcp-config <path>`` to the claude
    CLI.  Tool calls happen inside the claude subprocess; the provider returns
    the final assistant text directly — ``tool_calls`` on the ChatResponse is
    always empty.  Requires ``options["persona_dir"]`` so the spawned server
    knows which persona directory to load.

    When ``tools`` is None, the text ``-p`` path applies. Multi-turn history
    is encoded as JSONL context to avoid role-label leakage.
    """

    def __init__(
        self,
        model: str = "sonnet",
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", self._model]
        if system is not None:
            cmd.extend(["--system-prompt", system])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ClaudeCliProvider: subprocess timed out after {self._timeout}s"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ClaudeCliProvider failed (exit {result.returncode}): "
                f"{_claude_failure_detail(result)}"
            )

        try:
            payload = json.loads(result.stdout)
            return str(payload["result"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"ClaudeCliProvider: unexpected output format: {result.stdout[:200]!r}"
            ) from exc

    def name(self) -> str:
        return f"claude-cli:{self._model}"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Flatten messages into Claude CLI; route through MCP when tools given.

        When ``tools`` is None:
          - Sends the current user turn, plus multi-turn JSONL context when
            needed, via ``-p``.
          - Returns ChatResponse(content=<text>, tool_calls=()).

        When ``tools`` is provided:
          - Requires ``options["persona_dir"]`` to point the MCP server at the
            active persona.
          - Writes a temp mcp.json, calls claude with --mcp-config, returns
            the final assistant text. Tool calls happen inside the claude
            subprocess; tool_calls on the response is always empty.

        Raises
        ------
        ProviderError("claude_cli_timeout", ...)
        ProviderError("claude_cli_exit", ...)
        ProviderError("claude_cli_parse", ...)
        ProviderError("mcp_unavailable", ...)
            When tools is non-None and options["persona_dir"] is missing,
            or when the mcp SDK is not importable.
        ProviderError("claude_cli_setup", ...)
            When the temp config file write fails.
        """
        system_prompt: str | None = None
        conversation_messages: list[ChatMessage] = []
        for msg in messages:
            if msg.role == "system" and system_prompt is None:
                # content_text() flattens both legacy str content and the
                # tuple[ContentBlock, ...] multimodal shape uniformly.
                system_prompt = msg.content_text()
            else:
                conversation_messages.append(msg)

        # If any conversation message carries an ImageBlock, route through
        # the stream-json input path so claude actually sees the pixels.
        # The legacy text path stays the default for text-only turns —
        # cheaper, no JSON-stream framing overhead.
        has_images = any(_message_has_image(m) for m in conversation_messages)
        if has_images:
            persona_dir_str = (options or {}).get("persona_dir")
            if not persona_dir_str:
                raise ProviderError(
                    "image_passthrough_unavailable",
                    "image-bearing turns require options['persona_dir'] "
                    "to resolve <persona_dir>/images/<sha>.<ext>",
                )
            return self._chat_with_images(
                conversation_messages=conversation_messages,
                system_prompt=system_prompt,
                persona_dir=Path(persona_dir_str),
                tools_enabled=bool(tools),
            )

        flat_prompt = _format_claude_print_prompt(conversation_messages)

        if tools:
            persona_dir_str = (options or {}).get("persona_dir")
            if not persona_dir_str:
                raise ProviderError(
                    "mcp_unavailable",
                    "tool-calling via MCP requires options['persona_dir']",
                )
            return self._chat_with_mcp_tools(
                flat_prompt=flat_prompt,
                system_prompt=system_prompt,
                persona_dir=Path(persona_dir_str),
            )

        # Text path for turns without image blocks. Multi-turn history is
        # formatted as JSONL context data instead of a script with canonical
        # User:/Assistant: labels, because those labels can leak into replies.
        cmd = ["claude", "-p", flat_prompt, "--output-format", "json", "--model", self._model]
        if system_prompt is not None:
            cmd.extend(["--system-prompt", system_prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                "claude_cli_timeout",
                f"subprocess timed out after {self._timeout}s",
            ) from exc

        if result.returncode != 0:
            raise ProviderError(
                "claude_cli_exit",
                f"exit {result.returncode}: {_claude_failure_detail(result)}",
            )

        try:
            payload = json.loads(result.stdout)
            content = str(payload["result"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderError(
                "claude_cli_parse",
                f"unexpected output format: {result.stdout[:200]!r}",
            ) from exc

        return ChatResponse(
            content=_truncate_at_role_leak(content),
            tool_calls=(),
            raw=None,
        )

    def _chat_with_images(
        self,
        conversation_messages: list[ChatMessage],
        system_prompt: str | None,
        persona_dir: Path,
        tools_enabled: bool,
    ) -> ChatResponse:
        """Multimodal chat path — pipes Anthropic-shaped messages via stream-json.

        Why: claude CLI has no native ``--image`` flag. ``--input-format
        stream-json`` does accept SDK-shape messages with content blocks
        including ``{type: image, source: {type: base64, ...}}`` (verified
        2026-05-07). This path is reserved for turns that actually carry
        ImageBlocks; pure text turns continue through the legacy ``-p``
        path which avoids the JSON-stream framing overhead.

        Tool-calling: when ``tools_enabled`` is True we still write a
        temp mcp.json + ``--mcp-config`` + ``--allowedTools`` so MCP tool
        calls inside the claude subprocess work the same as in the
        text-only path. Stream-json input is orthogonal to MCP.

        Persisted history: only the LAST user turn is sent as a
        multimodal block list — earlier turns are flattened into the
        system prompt because stream-json sends one user message at a
        time. This matches how the legacy text path collapses prior
        turns into the prompt body.
        """
        # Split: history (everything but the last user turn) goes into
        # system prompt as a flat transcript; the final user turn goes
        # through stream-json as Anthropic content blocks.
        if not conversation_messages:
            raise ProviderError(
                "image_passthrough_unavailable",
                "no conversation messages to send",
            )
        last_user = conversation_messages[-1]
        if last_user.role != "user":
            raise ProviderError(
                "image_passthrough_unavailable",
                f"last message must be role=user; got {last_user.role!r}",
            )
        history = conversation_messages[:-1]

        # Compose the final system prompt: original system + JSONL context
        # data for prior turns. Multi-turn image conversations work; what
        # changes is that earlier images render as [image: <sha[:8]>]
        # text markers in the history, not as visible blocks. We deliberately
        # avoid ``User:`` / ``Assistant:`` transcript labels here too; those
        # labels prime Claude to continue the script in its own reply.
        history_text = ""
        if history:
            history_text = _format_claude_context_block(history, includes_latest_user=False)
        full_system: str | None = system_prompt
        if history_text:
            full_system = f"{system_prompt}\n\n{history_text}" if system_prompt else history_text

        # Build the SDK-shape user message frame.
        user_frame = _build_stream_json_user_message(last_user, persona_dir)
        stdin_payload = json.dumps(user_frame, ensure_ascii=False) + "\n"

        cmd = [
            "claude",
            "--print",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self._model,
        ]
        if full_system is not None:
            cmd.extend(["--system-prompt", full_system])

        # Set up MCP if tools are enabled — same shape as the
        # legacy-text tool path so the brain's tools remain available
        # for image-bearing turns.
        tmp_mcp_path: str | None = None
        request_id = uuid.uuid4().hex
        env_overrides = {**os.environ, "NELL_MCP_AUDIT_REQUEST_ID": request_id}
        audit_offset_before = 0
        audit_log_path = persona_dir / "tool_invocations.log.jsonl"
        if tools_enabled:
            try:
                import mcp  # noqa: F401
            except ImportError as exc:
                raise ProviderError(
                    "mcp_unavailable",
                    "the 'mcp' SDK is required for image-passthrough + tools",
                ) from exc
            mcp_config = {
                "mcpServers": {
                    "brain-tools": {
                        "command": sys.executable,
                        "args": [
                            "-m",
                            "brain.mcp_server",
                            "--persona-dir",
                            str(persona_dir),
                        ],
                        "env": {"NELL_MCP_AUDIT_REQUEST_ID": request_id},
                    }
                }
            }
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8",
                ) as tmp:
                    json.dump(mcp_config, tmp)
                    tmp_mcp_path = tmp.name
            except OSError as exc:
                raise ProviderError(
                    "claude_cli_setup",
                    f"failed to write temp mcp.json: {exc}",
                ) from exc
            from brain.tools import NELL_TOOL_NAMES

            allowed_mcp = [f"mcp__brain-tools__{n}" for n in NELL_TOOL_NAMES]
            cmd.extend(["--mcp-config", tmp_mcp_path])
            cmd.extend(["--allowedTools", *allowed_mcp])
            try:
                audit_offset_before = audit_log_path.stat().st_size
            except FileNotFoundError:
                audit_offset_before = 0

        try:
            try:
                result = subprocess.run(
                    cmd,
                    input=stdin_payload,
                    capture_output=True,
                    text=True,
                encoding="utf-8",
                errors="replace",
                    timeout=self._timeout,
                    env=env_overrides,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ProviderError(
                    "claude_cli_timeout",
                    f"image-passthrough subprocess timed out after {self._timeout}s",
                ) from exc

            if result.returncode != 0:
                raise ProviderError(
                    "claude_cli_exit",
                    f"exit {result.returncode}: {_claude_failure_detail(result)}",
                )

            content = _parse_stream_json_result(result.stdout)
            dispatched: tuple[dict[str, Any], ...] = ()
            if tools_enabled:
                dispatched = tuple(
                    _read_audit_lines_since(
                        audit_log_path, audit_offset_before, request_id=request_id
                    )
                )
            return ChatResponse(
                content=_truncate_at_role_leak(content),
                tool_calls=(),
                dispatched_invocations=dispatched,
                raw=None,
            )
        finally:
            if tmp_mcp_path:
                try:
                    os.unlink(tmp_mcp_path)
                except OSError:
                    pass

    def _chat_with_mcp_tools(
        self,
        flat_prompt: str,
        system_prompt: str | None,
        persona_dir: Path,
    ) -> ChatResponse:
        """Tool-calling path: claude with --mcp-config pointing at brain.mcp_server.

        The mcp SDK is only imported here — keeps the legacy text path
        usable on systems without the SDK installed.

        Telemetry: snapshots the persona's audit log size before invoking
        the claude subprocess; reads any newly-appended lines afterward
        and surfaces them as ChatResponse.dispatched_invocations. Per-
        session /chat is serialized via in_flight_locks so no other
        writer interleaves into the snapshot window. Tools dispatched
        here have ALREADY run inside the subprocess — run_tool_loop
        must not re-dispatch them.
        """
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ProviderError(
                "mcp_unavailable",
                "the 'mcp' SDK is required for the Claude tool-calling path. "
                "pip install 'mcp>=1.0.0,<2.0.0'",
            ) from exc

        request_id = uuid.uuid4().hex
        config = {
            "mcpServers": {
                "brain-tools": {
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "brain.mcp_server",
                        "--persona-dir",
                        str(persona_dir),
                    ],
                    "env": {"NELL_MCP_AUDIT_REQUEST_ID": request_id},
                }
            }
        }

        # Snapshot audit log size BEFORE subprocess so we can read only
        # the lines this subprocess appends.
        audit_log_path = persona_dir / "tool_invocations.log.jsonl"
        try:
            audit_offset_before = audit_log_path.stat().st_size
        except FileNotFoundError:
            audit_offset_before = 0

        tmp_path: str | None = None
        try:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    json.dump(config, tmp)
                    tmp_path = tmp.name
            except OSError as exc:
                raise ProviderError(
                    "claude_cli_setup",
                    f"failed to write temp mcp.json: {exc}",
                ) from exc

            # Build the list of allowed MCP tool names for --allowedTools.
            # Claude CLI blocks MCP tool calls in non-interactive (-p) mode
            # unless each tool is explicitly pre-approved.  The MCP server
            # name in mcp.json is "brain-tools", so Claude registers tools
            # as "mcp__brain-tools__<name>". We allow all registered
            # brain-tools so the LLM can call them without a permission
            # prompt.
            from brain.tools import NELL_TOOL_NAMES  # local import — avoids circular

            allowed_mcp = [f"mcp__brain-tools__{n}" for n in NELL_TOOL_NAMES]
            cmd = ["claude", "-p", flat_prompt, "--output-format", "json", "--model", self._model]
            if system_prompt is not None:
                cmd.extend(["--system-prompt", system_prompt])
            cmd.extend(["--mcp-config", tmp_path])
            cmd.extend(["--allowedTools", *allowed_mcp])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                encoding="utf-8",
                errors="replace",
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ProviderError(
                    "claude_cli_timeout",
                    f"subprocess timed out after {self._timeout}s",
                ) from exc

            if result.returncode != 0:
                raise ProviderError(
                    "claude_cli_exit",
                    f"exit {result.returncode}: {_claude_failure_detail(result)}",
                )

            try:
                payload = json.loads(result.stdout)
                content = str(payload["result"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ProviderError(
                    "claude_cli_parse",
                    f"unexpected output format: {result.stdout[:200]!r}",
                ) from exc

            dispatched = _read_audit_lines_since(
                audit_log_path,
                audit_offset_before,
                request_id=request_id,
            )
            return ChatResponse(
                content=_truncate_at_role_leak(content),
                tool_calls=(),
                dispatched_invocations=tuple(dispatched),
                raw=None,
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _read_audit_lines_since(
    audit_log_path: Path,
    offset: int,
    *,
    request_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read audit log lines newly appended after `offset` bytes.

    Returns a list of invocation records in the engine's tool_invocations
    shape: ``{name, arguments, result_summary, error?}``. Malformed lines
    are skipped silently — telemetry should never break a chat response.
    """
    try:
        with audit_log_path.open("rb") as fh:
            fh.seek(offset)
            new_bytes = fh.read()
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("MCP audit telemetry read failed for %s: %s", audit_log_path, exc)
        return []

    records: list[dict[str, Any]] = []
    malformed = 0
    for raw_line in new_bytes.splitlines():
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if request_id is not None:
            entry_request_id = entry.get("request_id")
            # Records produced by current code carry request_id and can be
            # strictly correlated. Legacy no-id records are still admitted so
            # old tests/logs remain readable, but records from another current
            # request are filtered out.
            if entry_request_id not in (None, request_id):
                continue
        record: dict[str, Any] = {
            "name": entry.get("name", "?"),
            "arguments": entry.get("arguments", {}),
            "result_summary": entry.get("result_summary", ""),
        }
        if entry.get("error"):
            record["error"] = entry["error"]
        records.append(record)
    if malformed:
        logger.warning(
            "MCP audit telemetry skipped %d malformed line(s) from %s",
            malformed,
            audit_log_path,
        )
    return records


# ---------------------------------------------------------------------------
# ClaudeCliProvider — prompt/context formatting helpers
# ---------------------------------------------------------------------------


_CLAUDE_SAFE_SPEAKERS = {
    "user": "human",
    "assistant": "companion",
    "tool": "tool_result",
}


# Lines beginning with a known role label followed by ': ' indicate the
# model has stopped replying and started scripting the next turn — a
# common failure on creative scenes where the model treats its reply
# as multi-turn fiction. We truncate at the first such marker.
#
# Conservative: only matches at the start of a line (after \n) and only
# the canonical role-label set. Lowercase variants included because the
# model occasionally emits 'user:' (Nell's voice is lowercase-leaning).
# We do NOT include persona-specific names like 'Hana:' / 'Nell:' here
# because Hana legitimately quotes those in fiction.
_ROLE_LEAK_PATTERN = re.compile(
    r"\n(?:User|user|Human|human|Assistant|assistant):\s",
)


def _truncate_at_role_leak(text: str) -> str:
    """Strip everything from the first role-label line onward.

    Idempotent and safe: when no leak is present, returns ``text``
    unchanged. When the model overruns into a hypothetical multi-turn
    script, returns just the prefix up to (but not including) the
    overrun. The trailing newline before the cut is preserved so the
    reply still ends on a clean line.
    """
    match = _ROLE_LEAK_PATTERN.search(text)
    if match is None:
        return text
    return text[: match.start()].rstrip()


def _format_claude_print_prompt(messages: list[ChatMessage]) -> str:
    """Format messages for ``claude -p`` without transcript role labels.

    A single text turn stays verbatim to preserve the simple generation shape.
    Multi-turn history is encoded as JSONL data with non-canonical speaker
    names. The previous ``User:`` / ``Assistant:`` script shape primed Claude
    to continue that transcript, which sometimes leaked role labels into
    Nell's visible reply.
    """
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0].content_text()
    return _format_claude_context_block(messages, includes_latest_user=True)


def _format_claude_context_block(
    messages: list[ChatMessage],
    *,
    includes_latest_user: bool,
) -> str:
    """Return a JSONL context block for Claude CLI prompt/system text.

    The wording intentionally avoids the canonical ``User:`` and
    ``Assistant:`` delimiters. JSON-string encoding also keeps embedded
    newlines or user-supplied delimiter-looking text inside the data field
    instead of creating new transcript lines.
    """
    if includes_latest_user:
        instruction = (
            "Answer the final human entry directly in the companion's voice. "
            "Do not prefix the reply with any speaker name, role label, JSON, "
            "or transcript marker."
        )
    else:
        instruction = (
            "Use this only as context for the next human message. Do not "
            "prefix the reply with any speaker name, role label, JSON, or "
            "transcript marker."
        )

    lines = [
        "Conversation context is encoded below as JSONL data, not as a transcript to continue.",
        instruction,
        "",
    ]
    lines.extend(_claude_context_jsonl_lines(messages))
    return "\n".join(lines)


def _claude_context_jsonl_lines(messages: list[ChatMessage]) -> Iterator[str]:
    """Yield one JSON object per chat turn using leak-resistant speaker names."""
    for msg in messages:
        record: dict[str, Any] = {
            "speaker": _CLAUDE_SAFE_SPEAKERS.get(msg.role, msg.role),
            "text": msg.content_text(),
        }
        if msg.tool_call_id:
            record["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            record["tool_calls"] = [tc.to_dict() for tc in msg.tool_calls]
        yield json.dumps(record, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ClaudeCliProvider — image passthrough helpers
# ---------------------------------------------------------------------------


def _message_has_image(msg: ChatMessage) -> bool:
    """True if ``msg.content`` carries any ImageBlock."""
    if isinstance(msg.content, str):
        return False
    return any(isinstance(b, ImageBlock) for b in msg.content)


def _build_stream_json_user_message(
    msg: ChatMessage,
    persona_dir: Path,
) -> dict[str, Any]:
    """Convert a ChatMessage with image blocks into Anthropic-shaped JSON.

    Reads image bytes from disk, base64-encodes inline, and emits the
    SDK-shape envelope claude --input-format stream-json expects:

        {"type": "user", "message": {"role": "user", "content": [...]}}

    where each content block is either ``{type: text, text}`` or
    ``{type: image, source: {type: base64, media_type, data}}``.

    Image-block bytes are read via ``brain.images.read_image_bytes``;
    missing files surface as ProviderError("image_missing", ...) so the
    caller can decide whether to skip or fail.
    """
    import base64

    from brain.images import read_image_bytes

    blocks_out: list[dict[str, Any]] = []
    for block in msg.content_blocks():
        if isinstance(block, TextBlock):
            if block.text:
                blocks_out.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            try:
                raw = read_image_bytes(persona_dir, block.image_sha, block.media_type)
            except FileNotFoundError as exc:
                raise ProviderError(
                    "image_missing",
                    f"image_sha={block.image_sha[:8]} not on disk: {exc}",
                ) from exc
            blocks_out.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media_type,
                        "data": base64.b64encode(raw).decode("ascii"),
                    },
                }
            )
    return {
        "type": "user",
        "message": {"role": msg.role, "content": blocks_out},
    }


def _parse_stream_json_result(stdout: str) -> str:
    """Extract the final assistant reply from a claude stream-json output.

    The stream emits a sequence of JSON-line events; the canonical reply
    text lives on the ``{"type": "result", "result": "..."}`` frame
    emitted last. If no result frame is present, fall back to
    concatenating any ``{"type":"assistant"}`` content with type "text".
    """
    result_text: str | None = None
    assistant_chunks: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        ftype = frame.get("type")
        if ftype == "result" and "result" in frame:
            result_text = str(frame["result"])
        elif ftype == "assistant":
            content = frame.get("message", {}).get("content") or []
            for block in content:
                if block.get("type") == "text":
                    assistant_chunks.append(str(block.get("text", "")))
    if result_text is not None:
        return result_text
    if assistant_chunks:
        return "".join(assistant_chunks)
    raise ProviderError(
        "claude_cli_parse",
        f"no result frame in stream-json output: {stdout[:300]!r}",
    )


# ---------------------------------------------------------------------------
# OllamaProvider — httpx-based, full tool-call support
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Local Ollama integration over httpx.

    Full port of OG NellBrain's OllamaProvider (nell_bridge_providers.py).

    Default model: huihui_ai/qwen2.5-abliterated:7b — uncensored Qwen2.5
    abliterated variant.  Same base architecture as Nell's nell-dpo fine-tune,
    light enough for most local hardware (~4.7GB), works well with the brain's
    emotional + creative tone.  Users can override via OllamaProvider(model="…").

    Streaming
    ---------
    ``chat_stream()`` POSTs with ``stream=True`` and yields each content
    chunk as Ollama emits it. The bridge WS handler (or any other
    streaming consumer) can pipe these straight to the client without
    waiting for the full reply. Tool-calling is intentionally not on
    the streaming path — pass ``tools=`` through :meth:`chat` instead,
    which returns a structured ChatResponse with parsed tool_calls.
    """

    def __init__(
        self,
        model: str = "huihui_ai/qwen2.5-abliterated:7b",
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def name(self) -> str:
        return f"ollama:{self._model}"

    def healthy(self) -> bool:
        """GET /api/tags — True if Ollama is reachable and responds 200."""
        try:
            r = httpx.get(f"{self._host}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.RequestError:
            return False

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """POST to /api/chat and parse the structured response.

        Raises
        ------
        ProviderError("ollama_http", ...)
            HTTP error response from Ollama (4xx / 5xx).
        ProviderError("ollama_request", ...)
            Network-level failure (connection refused, DNS, etc.).
        ProviderError("ollama_parse", ...)
            Response body is not valid JSON or missing expected fields.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if options:
            generation_options = {
                key: value
                for key, value in options.items()
                if key not in _PROVIDER_CONTEXT_OPTION_KEYS
            }
            if generation_options:
                payload["options"] = generation_options

        url = f"{self._host}/api/chat"
        try:
            resp = httpx.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                "ollama_http",
                f"{exc.response.status_code}: {exc.response.text[:200]}",
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError("ollama_request", str(exc)) from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError("ollama_parse", f"invalid json: {exc}") from exc

        msg = data.get("message") or {}
        content: str = msg.get("content", "") or ""
        raw_tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []

        parsed_tool_calls: list[ToolCall] = []
        for tc_dict in raw_tool_calls:
            try:
                parsed_tool_calls.append(ToolCall.from_provider_dict(tc_dict))
            except ValueError as exc:
                raise ProviderError(
                    "ollama_parse",
                    f"malformed tool_call in response: {exc}",
                ) from exc

        return ChatResponse(
            content=_truncate_at_role_leak(content),
            tool_calls=tuple(parsed_tool_calls),
            raw=data,
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream content chunks from /api/chat as Ollama emits them.

        Each yielded value is a str fragment of the assistant's reply.
        The iterator exhausts when Ollama signals ``done=True``. Joining
        all yielded chunks reproduces the same content :meth:`chat`
        returns for the same prompt.

        Tool-calling is not supported here — Ollama's tool_calls arrive
        on the final stream frame, which complicates the perceived-typing
        UX this method exists to power. Callers needing tools should use
        :meth:`chat` and word-chunk the result client-side, or call
        :meth:`chat` first then stream a follow-up turn.

        Parameters
        ----------
        messages:
            Conversation history in chronological order.
        options:
            Provider-specific generation options. Reserved keys
            (``persona_dir``) are stripped before forwarding.

        Yields
        ------
        str
            Each content chunk Ollama emits. Empty chunks are skipped.

        Raises
        ------
        ProviderError("ollama_http", ...)
            HTTP error response from Ollama (4xx / 5xx).
        ProviderError("ollama_request", ...)
            Network-level failure (connection refused, DNS, etc.).
        ProviderError("ollama_parse", ...)
            A streamed line could not be JSON-decoded.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        if options:
            generation_options = {
                key: value
                for key, value in options.items()
                if key not in _PROVIDER_CONTEXT_OPTION_KEYS
            }
            if generation_options:
                payload["options"] = generation_options

        url = f"{self._host}/api/chat"
        try:
            with httpx.stream(
                "POST", url, json=payload, timeout=self._timeout
            ) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = exc.response.read().decode("utf-8", errors="replace")
                    raise ProviderError(
                        "ollama_http",
                        f"{exc.response.status_code}: {body[:200]}",
                    ) from exc
                for raw_line in resp.iter_lines():
                    line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(
                            "ollama_parse",
                            f"invalid streaming line json: {exc}",
                        ) from exc
                    msg = frame.get("message") or {}
                    chunk = msg.get("content")
                    if chunk:
                        yield chunk
                    if frame.get("done"):
                        return
        except httpx.RequestError as exc:
            raise ProviderError("ollama_request", str(exc)) from exc

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Single-turn text generation.

        Implemented by delegating to chat() so all request/response logic
        lives in one place.  This is the first working Ollama generate() path
        in companion-emergence — previously this raised NotImplementedError.
        """
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        response = self.chat(messages)
        return response.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider(name: str) -> LLMProvider:
    """Resolve a provider identifier to an instance.

    Raises ValueError on unknown name.
    """
    if name == "fake":
        return FakeProvider()
    if name == "claude-cli":
        return ClaudeCliProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown provider: {name!r}")
