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
import subprocess
from abc import ABC, abstractmethod
from typing import Any

import httpx

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 300


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
      - Remaining messages are serialised as a "User: …\\nAssistant: …" script
        ending at the final user turn, passed via ``-p``.

    Tool-calling via --json-schema
    ------------------------------
    When ``tools`` is provided, the provider augments the system prompt with
    instructions about available tools and passes a ``--json-schema`` flag
    enforcing a discriminated-union response: either ``{"reply": "..."}`` or
    ``{"tool_calls": [{"name": "...", "arguments": {...}}, ...]}``.

    Claude's ``--output-format json`` with ``--json-schema`` returns the
    structured output inside ``payload["structured_output"]``.  The provider
    parses this and returns either a plain-text ChatResponse (reply path) or
    a ChatResponse with tool_calls populated (tool-calling path).

    When ``tools`` is None, the legacy text-flattening path applies — behaviour
    is unchanged from before SP-3.
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
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ClaudeCliProvider: subprocess timed out after {self._timeout}s"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ClaudeCliProvider failed (exit {result.returncode}): {result.stderr.strip()}"
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
        """Flatten messages into Claude CLI and optionally enforce tool-calling schema.

        When ``tools`` is None:
          - Flattens messages into "User: ...\\nAssistant: ..." script via -p.
          - Returns ChatResponse(content=<text>, tool_calls=()).

        When ``tools`` is provided:
          - Augments the system prompt with tool instructions.
          - Builds a discriminated-union ``--json-schema`` that forces Claude to
            respond with either ``{"reply": "..."}`` or
            ``{"tool_calls": [{"name": "...", "arguments": {...}}]}``.
          - Parses ``structured_output`` from the JSON response and returns the
            appropriate ChatResponse.

        Raises
        ------
        ProviderError("claude_cli_timeout", ...)
            If the subprocess exceeds the configured timeout.
        ProviderError("claude_cli_exit", ...)
            If the subprocess exits with a non-zero return code.
        ProviderError("claude_cli_parse", ...)
            If the JSON output cannot be parsed or lacks expected keys.
        ProviderError("claude_schema", ...)
            If the schema-constrained invocation returns an unexpected shape.
        """
        system_prompt: str | None = None
        conversation_messages: list[ChatMessage] = []

        for msg in messages:
            if msg.role == "system" and system_prompt is None:
                system_prompt = msg.content
            else:
                conversation_messages.append(msg)

        # Build a flat conversation script for the -p flag.
        # Format: "User: ...\nAssistant: ...\nUser: ..." ending on user turn.
        if not conversation_messages:
            flat_prompt = ""
        elif len(conversation_messages) == 1:
            flat_prompt = conversation_messages[0].content
        else:
            parts: list[str] = []
            role_labels = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
            for msg in conversation_messages:
                label = role_labels.get(msg.role, msg.role.capitalize())
                parts.append(f"{label}: {msg.content}")
            flat_prompt = "\n".join(parts)

        if tools:
            return self._chat_with_tools(
                flat_prompt=flat_prompt,
                system_prompt=system_prompt,
                tools=tools,
            )

        # ── Legacy text path (no tools) ──────────────────────────────────────
        cmd = ["claude", "-p", flat_prompt, "--output-format", "json", "--model", self._model]
        if system_prompt is not None:
            cmd.extend(["--system-prompt", system_prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
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
                f"exit {result.returncode}: {result.stderr.strip()}",
            )

        try:
            payload = json.loads(result.stdout)
            content = str(payload["result"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderError(
                "claude_cli_parse",
                f"unexpected output format: {result.stdout[:200]!r}",
            ) from exc

        return ChatResponse(content=content, tool_calls=(), raw=None)

    def _build_tool_call_schema(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a discriminated-union JSON schema for tool-calling responses.

        Claude must respond with EITHER:
          {"reply": "<text>"}          — final answer, no tool needed
          {"tool_calls": [...]}        — one or more tool invocations

        The ``name`` field in each tool_call is an enum of known tool names
        so Claude cannot hallucinate an unknown tool.
        """
        tool_names = [t["name"] for t in tools if "name" in t]
        return {
            "type": "object",
            "oneOf": [
                {
                    "properties": {
                        "reply": {"type": "string"},
                    },
                    "required": ["reply"],
                    "additionalProperties": False,
                },
                {
                    "properties": {
                        "tool_calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "enum": tool_names,
                                    },
                                    "arguments": {"type": "object"},
                                },
                                "required": ["name", "arguments"],
                            },
                            "minItems": 1,
                        }
                    },
                    "required": ["tool_calls"],
                    "additionalProperties": False,
                },
            ],
        }

    def _build_tool_system_addendum(self, tools: list[dict[str, Any]]) -> str:
        """Build the system-prompt addendum describing available tools."""
        lines = ["You have access to these tools:"]
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "no description")
            lines.append(f"  - {name}: {desc}")
        lines.append(
            "\nRespond with EITHER a final reply (use the 'reply' field) OR a "
            "tool_calls array if you need to query state first. Do not mix both."
        )
        return "\n".join(lines)

    def _chat_with_tools(
        self,
        flat_prompt: str,
        system_prompt: str | None,
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        """Inner path: invoke Claude CLI with --json-schema for tool-calling.

        Returns a ChatResponse with either content (reply path) or tool_calls
        populated (tool-call path).
        """
        import uuid as _uuid

        schema = self._build_tool_call_schema(tools)
        tool_addendum = self._build_tool_system_addendum(tools)

        # Combine system prompt + tool instructions
        if system_prompt:
            full_system = f"{system_prompt}\n\n{tool_addendum}"
        else:
            full_system = tool_addendum

        cmd = [
            "claude",
            "-p",
            flat_prompt,
            "--output-format",
            "json",
            "--model",
            self._model,
            "--json-schema",
            json.dumps(schema),
            "--system-prompt",
            full_system,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                "claude_cli_timeout",
                f"subprocess timed out after {self._timeout}s (tools path)",
            ) from exc

        if result.returncode != 0:
            raise ProviderError(
                "claude_cli_exit",
                f"exit {result.returncode}: {result.stderr.strip()} (tools path)",
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "claude_cli_parse",
                f"tools path: non-JSON output: {result.stdout[:200]!r}",
            ) from exc

        # Claude returns structured output under the "structured_output" key
        # when --json-schema is honored. In practice, a sufficiently rich
        # system prompt (e.g. a persona's voice.md) can outweigh the schema
        # enforcement and Claude returns a plain text reply in `result`
        # instead. When that happens, treat it as a no-tool-call reply rather
        # than erroring — Claude correctly answered the user; it just chose
        # not to enforce the protocol envelope. This matches how the chat
        # engine's tool loop interprets ChatResponse(content=..., tool_calls=())
        # as "final reply, no tools needed."
        structured = payload.get("structured_output")
        if structured is None:
            plain_result = payload.get("result")
            if isinstance(plain_result, str) and plain_result.strip():
                logger.info(
                    "ClaudeCliProvider: response missing structured_output; "
                    "treating result as plain reply (Claude went off-schema, "
                    "likely due to rich system prompt). %d chars returned.",
                    len(plain_result),
                )
                return ChatResponse(
                    content=plain_result,
                    tool_calls=(),
                    raw=payload,
                )
            raise ProviderError(
                "claude_schema",
                f"tools path: missing 'structured_output' AND no usable 'result' in response: {result.stdout[:300]!r}",
            )

        # Reply path
        if "reply" in structured:
            return ChatResponse(
                content=str(structured["reply"]),
                tool_calls=(),
                raw=payload,
            )

        # Tool-calls path
        if "tool_calls" in structured:
            parsed_tool_calls: list[ToolCall] = []
            for tc in structured["tool_calls"]:
                call_id = str(_uuid.uuid4())
                try:
                    parsed_tool_calls.append(
                        ToolCall(
                            id=call_id,
                            name=str(tc["name"]),
                            arguments=dict(tc.get("arguments") or {}),
                        )
                    )
                except (KeyError, TypeError) as exc:
                    raise ProviderError(
                        "claude_schema",
                        f"tools path: malformed tool_call entry {tc!r}: {exc}",
                    ) from exc
            return ChatResponse(
                content="",
                tool_calls=tuple(parsed_tool_calls),
                raw=payload,
            )

        raise ProviderError(
            "claude_schema",
            f"tools path: structured_output has neither 'reply' nor 'tool_calls': {structured!r}",
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
    # TODO(streaming): chat_stream() — Phase 6.5 scope.  The non-streaming
    # chat() here is sufficient for all SP-1 through SP-5 use-cases.  When
    # Phase 6.5 lands, add a chat_stream() that POSTs with stream=True and
    # yields token chunks.
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
            payload["options"] = options

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
            content=content,
            tool_calls=tuple(parsed_tool_calls),
            raw=data,
        )

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
