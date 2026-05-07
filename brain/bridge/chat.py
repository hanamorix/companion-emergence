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
import re
from dataclasses import dataclass, field
from typing import Any, Literal

ChatRole = Literal["system", "user", "assistant", "tool"]

_ALLOWED_MEDIA_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TextBlock:
    """Text segment inside a multimodal message."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImageBlock:
    """Image segment inside a multimodal message.

    Carries the sha-addressable filename so providers and ingest can read
    bytes from `<persona_dir>/images/<sha>.<ext>` without inlining them.
    media_type is the MIME type so callers can compute the file extension
    without re-sniffing.
    """

    image_sha: str
    media_type: str
    description: str | None = None
    type: Literal["image"] = "image"

    def __post_init__(self) -> None:
        if not _SHA256_HEX.fullmatch(self.image_sha):
            raise ValueError(
                f"ImageBlock.image_sha must be 64 lowercase hex chars, got {self.image_sha!r}"
            )
        if self.media_type not in _ALLOWED_MEDIA_TYPES:
            raise ValueError(
                f"ImageBlock.media_type must be one of {sorted(_ALLOWED_MEDIA_TYPES)}, got {self.media_type!r}"
            )


ContentBlock = TextBlock | ImageBlock


@dataclass(frozen=True)
class ChatMessage:
    """One turn in a multi-turn conversation.

    Attributes
    ----------
    role:
        Speaker: "system", "user", "assistant", or "tool".
    content:
        Either a plain str (legacy text-only path) or a tuple of
        ContentBlock entries (multimodal path). String input is auto-
        wrapped via ``__post_init__`` so existing callers keep working.
    tool_call_id:
        For role="tool" responses — the id of the ToolCall this answers.
        Must be None for all other roles.
    tool_calls:
        For role="assistant" turns that requested tool invocations.
        Empty tuple for all other roles.
    """

    role: ChatRole
    content: tuple[ContentBlock, ...] | str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    def content_text(self) -> str:
        """Flatten content to a single string.

        Plain-string content returns verbatim. Block-shaped content has
        TextBlocks contribute their text directly and ImageBlocks render
        as ``[image: <sha[:8]>]`` markers — short enough not to dominate
        token budget but unique enough for the extractor and downstream
        readers to know an image was present.
        """
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ImageBlock):
                parts.append(f"[image: {block.image_sha[:8]}]")
        return "\n".join(parts)

    def content_blocks(self) -> tuple[ContentBlock, ...]:
        """Return content as a tuple of blocks regardless of input shape.

        String content is wrapped into ``(TextBlock(text=str),)``. Use
        this when you need uniform iteration over blocks without caring
        which input shape the caller used.
        """
        if isinstance(self.content, str):
            return (TextBlock(text=self.content),)
        return self.content

    def to_dict(self) -> dict[str, Any]:
        """Serialise for provider transport.

        Pure-text messages serialise ``content`` as a plain string —
        that path is unchanged from the pre-multimodal shape so every
        existing provider and audit log keeps working.

        Block-shaped content with at least one ImageBlock serialises as
        an Anthropic-shaped list of typed dicts so multimodal-aware
        providers can pass it through verbatim.
        """
        d: dict[str, Any] = {"role": self.role}
        if isinstance(self.content, str):
            d["content"] = self.content
        elif all(isinstance(b, TextBlock) for b in self.content):
            d["content"] = "".join(
                b.text for b in self.content if isinstance(b, TextBlock)
            )
        else:
            d["content"] = [_block_to_dict(b) for b in self.content]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


def _block_to_dict(block: ContentBlock) -> dict[str, Any]:
    """Anthropic-shaped serialization for a content block."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    payload: dict[str, Any] = {
        "type": "image",
        "image_sha": block.image_sha,
        "media_type": block.media_type,
    }
    if block.description is not None:
        payload["description"] = block.description
    return payload


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
