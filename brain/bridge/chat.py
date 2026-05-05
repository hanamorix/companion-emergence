"""Structured message types for multi-turn LLM chat.

These frozen dataclasses are the lingua franca between providers and callers.
They are provider-agnostic — OllamaProvider, ClaudeCliProvider, and future
providers all produce / consume these same types.

ChatMessage   — one turn in a conversation (user / assistant / system / tool)
ToolCall      — a function invocation requested by the assistant
ChatResponse  — full provider response including optional tool-calls
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

ChatRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    """One turn in a multi-turn conversation.

    Attributes
    ----------
    role:
        Speaker: "system", "user", "assistant", or "tool".
    content:
        Text body of the message.
    tool_call_id:
        For role="tool" responses — the id of the ToolCall this answers.
        Must be None for all other roles.
    tool_calls:
        For role="assistant" turns that requested tool invocations.
        Empty tuple for all other roles.
    """

    role: ChatRole
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for provider transport.

        Omits optional fields that are empty/None so the payload stays clean.
        Converts the tool_calls tuple to a list of plain dicts for JSON
        serialisation.
        """
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


@dataclass(frozen=True)
class ToolCall:
    """A function invocation requested by the assistant.

    Attributes
    ----------
    id:
        Provider-assigned unique id for this specific call.
    name:
        The tool/function name (must match the schema registered with the provider).
    arguments:
        Parsed argument dict.  Providers return either pre-parsed dicts or
        JSON strings — from_provider_dict handles both.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Round-trip back to provider wire format."""
        return {
            "id": self.id,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    @classmethod
    def from_provider_dict(cls, d: dict[str, Any]) -> ToolCall:
        """Parse a provider-native tool-call dict.

        Handles both OpenAI/Ollama shape:
            {"id": "...", "function": {"name": "...", "arguments": "{...}" or {...}}}

        The ``arguments`` field may arrive as a pre-parsed dict (Claude) or as
        a JSON-encoded string (Ollama sometimes, older OpenAI shapes).

        Raises
        ------
        ValueError
            If the dict is missing required keys or arguments is not parseable.
        """
        try:
            call_id: str = d["id"]
            func = d["function"]
            name: str = func["name"]
            raw_args = func["arguments"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"ToolCall.from_provider_dict: missing required fields in {d!r}"
            ) from exc

        if isinstance(raw_args, dict):
            arguments = raw_args
        elif isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"ToolCall.from_provider_dict: cannot parse arguments JSON: {raw_args!r}"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"ToolCall.from_provider_dict: arguments must decode to a dict, got {type(arguments).__name__}"
                )
        else:
            raise ValueError(
                f"ToolCall.from_provider_dict: arguments must be dict or str, got {type(raw_args).__name__}"
            )

        return cls(id=call_id, name=name, arguments=arguments)


@dataclass(frozen=True)
class ChatResponse:
    """Full response from a multi-turn chat call.

    Attributes
    ----------
    content:
        Text reply from the assistant.  Empty string if the assistant only
        produced tool-calls and no prose.
    tool_calls:
        Tool invocations the assistant wants to make (empty tuple if none).
        These are NOT yet dispatched — the chat-engine's tool_loop runs
        them and feeds results back in a follow-up turn. OllamaProvider
        uses this path.
    dispatched_invocations:
        Tool invocations the provider ALREADY dispatched internally during
        this turn. ClaudeCliProvider's MCP path uses this — the claude
        subprocess invokes MCP tools via stdio and the provider surfaces
        the records here for telemetry. tool_loop must NOT re-dispatch
        these; they are observability data, not actionable requests.
        Each dict matches the engine invocation schema:
        ``{name, arguments, result_summary, error?}``.
    raw:
        Provider-native payload for debugging / logging.  None if the provider
        does not expose raw data (e.g. FakeProvider).
    """

    content: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    dispatched_invocations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    raw: dict[str, Any] | None = None
